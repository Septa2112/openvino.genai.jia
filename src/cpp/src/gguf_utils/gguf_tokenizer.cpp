// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_set>

#ifdef ENABLE_GGUF
#include "gguf.hpp"
#include "openvino/frontend/gguf/tokenizer_metadata.hpp"
#endif
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

using namespace ov::op;


constexpr int32_t MAX_LENGTH = 8192;
constexpr float VOCAB_SIZE_CACHE_PROPORTION = 0.2f;
constexpr int32_t MIN_CACHE_CAPACITY = 20'000;

namespace ov {
namespace genai {
bool is_gguf_model(const std::filesystem::path& file_path) {
    return file_path.extension() == ".gguf";
}

std::map<std::string, GGUFMetaData> tokenizer_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> tokenizer_config;

    const std::string prefix = "tokenizer.";
    for (const auto& [key, value] : metadata) {
        if (key.compare(0, prefix.size(), prefix) == 0) {
            size_t last_dot = key.find_last_of('.');
            // Extract the last part after "."
            std::string sub_key = (last_dot != std::string_view::npos) ? std::string(key.substr(last_dot + 1)) : key;
            tokenizer_config[sub_key] = value;
        }
    }

    return tokenizer_config;
}

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            FreeLibrary(static_cast<HMODULE>(h));
    });
#else
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string() + "\n" + dlerror());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            dlclose(h);
    });
#endif
}

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName) {
    if (!shared_object || !symbolName) {
        throw std::invalid_argument("Null shared object or symbol name.");
    }

#ifdef _WIN32
    HMODULE handle = static_cast<HMODULE>(shared_object.get());
    void* symbol = reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
    if (!symbol) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName));
    }
    return symbol;
#else
    void* handle = shared_object.get();
    // Clear existing errors
    dlerror();
    void* symbol = dlsym(handle, symbolName);
    const char* error = dlerror();
    if (error) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName) + "\n" + error);
    }
    return symbol;
#endif
}

ov::OutputVector add_ragged_dimension(const ov::OutputVector& inputs) {
    auto input_shape = std::make_shared<v3::ShapeOf>(inputs[0], element::i32);
    auto const_zero = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto const_one = std::make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto batch_size = std::make_shared<v8::Gather>(input_shape, const_zero, const_zero);
    auto batch_size_plus_one = std::make_shared<v1::Add>(batch_size, const_one);
    auto ragged_begins = std::make_shared<v4::Range>(const_zero, batch_size, const_one, element::i32)->output(0);
    auto ragged_ends = std::make_shared<v4::Range>(const_one, batch_size_plus_one, const_one, element::i32)->output(0);

    ov::OutputVector res = ov::OutputVector{ragged_begins, ragged_ends};
    res.insert(res.end(), inputs.begin(), inputs.end());
    return res;
}

bool is_special_token(int32_t token_type) {
    return token_type == 3 || token_type == 4;
}

std::string quote_meta(const std::string& str) {
    std::string result = "(";
    
    // Only ASCII punctuation needs escaping. Bytes of multi-byte UTF-8 characters (e.g. the fullwidth
    // bar and the lower one-eighth block in "<｜begin▁of▁sentence｜>") must stay together: a
    // backslash between them makes the pattern invalid UTF-8 and the token never matches.
    for (unsigned char c : str) {
        if (c < 0x80 && !std::isalnum(c) && c != '_') {
            result += '\\';
        }
        result += static_cast<char>(c);
    }
    result += ")";
    return result;
}

std::string join_special_tokens(const std::vector<std::string>& special_tokens) {
    std::ostringstream oss;
    for (size_t i = 0; i < special_tokens.size(); ++i) {
        if (i > 0)
            oss << "|";
        oss << quote_meta(special_tokens[i]);
    }
    return oss.str();
}

std::vector<std::string> get_split_regex(const std::string& pre) {
    // taken from
    // https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/src/llama-vocab.cpp#L279
    // TODO: complete for other archs
    // The Llama-3 pattern, shared by every pre-tokenizer name llama.cpp maps to it (and DBRX / Smaug).
    static const std::vector<std::string> llama3 = {
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
        "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    };
    // GPT-2's pattern, for the pre-tokenizer names llama.cpp maps to it.
    static const std::vector<std::string> gpt2 = {
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    };
    static std::unordered_map<std::string, std::vector<std::string>> regex_map = {
        {"gpt-2", gpt2},      {"phi-2", gpt2},       {"jina-es", gpt2},   {"jina-de", gpt2},  {"gigachat", gpt2},
        {"jina-v2-es", gpt2}, {"jina-v2-de", gpt2},  {"a.x-4.0", gpt2},   {"mellum", gpt2},   {"modern-bert", gpt2},
        {"mpt", gpt2},        {"olmo", gpt2},        {"jais", gpt2},      {"trillion", gpt2}, {"granite-docling", gpt2},
        // Gemma 4 runs BPE merges over the whole text; only newlines split it (merges never contain them).
        {"gemma4", {"[^\\n]+|[\\n]+"}},
        {"llama3", llama3},   {"llama-v3", llama3},  {"llama-bpe", llama3}, {"falcon3", llama3},
        {"falcon-h1", llama3}, {"pixtral", llama3},  {"midm-2.0", llama3},  {"lfm2", llama3},
        {"jina-v5-nano", llama3}, {"dbrx", llama3},  {"smaug-bpe", llama3},
        {"qwen2",
         {
             // original regex from tokenizer.json
             // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|
             // ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
             "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
         }},
        {"smollm",
         {
             "\\p{N}",
             "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
         }},
    };

    if (regex_map.count(pre)) {
        return regex_map.at(pre);
    }

    std::vector<std::string> default_regex_exprs = {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]",
    };

    return default_regex_exprs;
}

// Keyed by the same tokenizer.ggml.pre value as get_split_regex(): pre-types whose llama.cpp
// vocab defaults to add_bos_token=true.
static bool pre_tokenizer_adds_bos(const std::string& pre) {
    static const std::set<std::string> pre_add_bos = {
        // LLAMA_VOCAB_PRE_TYPE_LLAMA3 group
        "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0",
        "lfm2", "jina-v5-nano",
        // standalone
        "tekken", "chameleon",
    };
    return pre_add_bos.count(pre) > 0;
}

ov::OutputVector create_string_constant(const std::vector<std::string>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

ov::OutputVector create_string_constant(const std::vector<std::vector<uint8_t>>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

const std::unordered_map<std::string, uint8_t>& unicode_to_bytes() {
    static const std::unordered_map<std::string, uint8_t> map = []() {
        std::vector<uint8_t> bs;

        // Range: '!' (33) to '~' (126)
        for (uint8_t i = static_cast<uint8_t>('!'); i <= static_cast<uint8_t>('~'); ++i) {
            bs.push_back(i);
        }

        // Range: '¡' (161) to '¬' (172)
        for (uint8_t i = 0xA1; i <= 0xAC; ++i) {
            bs.push_back(i);
        }

        // Range: '®' (174) to 'ÿ' (255)
        for (int32_t i = 0xAE; i <= 0xFF; ++i) {
            bs.push_back(static_cast<uint8_t>(i));
        }

        std::vector<int32_t> cs;
        cs.reserve(bs.size());
        for (uint8_t byte : bs) {
            cs.push_back(static_cast<int32_t>(byte));
        }

        int32_t n = 0;
        for (int32_t b = 0; b < 256; ++b) {
            uint8_t byte = static_cast<uint8_t>(b);
            if (std::find(bs.begin(), bs.end(), byte) == bs.end()) {
                bs.push_back(byte);
                cs.push_back(256 + n);
                ++n;
            }
        }

        std::unordered_map<std::string, uint8_t> result;
        for (size_t i = 0; i < cs.size(); ++i) {
            int32_t cp = cs[i];
            std::string utf8_char;

            if (cp <= 0x7F) {
                utf8_char += static_cast<char>(cp);
            } else if (cp <= 0x7FF) {
                utf8_char += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                utf8_char += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                utf8_char += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
                utf8_char += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            }

            result[utf8_char] = bs[i];
        }

        return result;
    }();

    return map;
}

int utf8_char_length(unsigned char lead_byte) {
    if ((lead_byte & 0b10000000) == 0)
        return 1;  // 0xxxxxxx (ASCII)
    else if ((lead_byte & 0b11100000) == 0b11000000)
        return 2;  // 110xxxxx
    else if ((lead_byte & 0b11110000) == 0b11100000)
        return 3;  // 1110xxxx
    else if ((lead_byte & 0b11111000) == 0b11110000)
        return 4;  // 11110xxx
    else
        return -1;  // Invalid
}

std::vector<std::string> split_utf8_chars(const std::string& input) {
    std::vector<std::string> result;
    size_t i = 0;

    while (i < input.size()) {
        unsigned char lead = static_cast<unsigned char>(input[i]);
        int len = utf8_char_length(lead);
        if (len <= 0 || i + len > input.size()) {
            std::cerr << "Invalid UTF-8 sequence at byte index " << i << std::endl;
            break;  // Stop on error
        }
        OPENVINO_ASSERT(
            std::numeric_limits<size_t>::max() - i > len,
            "UTF-8 character length exceeds size_t limit at index ", i
        );
        result.emplace_back(input.substr(i, len));
        i += len;
    }

    return result;
}

std::vector<uint8_t> apply_unicode_to_bytes(const std::string& token) {
    const auto& bytes_encoder = unicode_to_bytes();

    std::vector<uint8_t> res{};
    bool return_original = false;
    auto unicode_chars = split_utf8_chars(token);

    for (const auto& uni_char : unicode_chars) {
        if (bytes_encoder.count(uni_char)) {
            res.push_back(bytes_encoder.at(uni_char));
        } else {
            return_original = true;
            break;
        }
    }

    if (return_original) {
        std::vector<uint8_t> bytes(token.begin(), token.end());
        return bytes;
    }
    return res;
}

std::vector<std::vector<uint8_t>> parse_bbpe_vocab(const std::vector<std::string>& vocab, bool byte_encode = true) {
    std::vector<std::vector<uint8_t>> res;
    for (const auto& token : vocab) {
        // gpt2 uses GPT-2 unicode->byte decoding; gemma4's raw-UTF8 vocab is taken verbatim.
        res.push_back(byte_encode ? apply_unicode_to_bytes(token)
                                  : std::vector<uint8_t>(token.begin(), token.end()));
    }
    return res;
}

// Gemma 4: BPE over raw UTF-8 (no GPT-2 byte-to-unicode mapping); spaces are "▁" in the vocabulary.
std::vector<uint8_t> raw_bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

// What the detokenizer emits for a piece: "▁" is a space, and "<0xHH>" is the byte itself.
std::vector<uint8_t> decoded_piece(const std::string& piece) {
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>' && std::isxdigit(static_cast<unsigned char>(piece[3])) &&
        std::isxdigit(static_cast<unsigned char>(piece[4]))) {
        return {static_cast<uint8_t>(std::stoi(piece.substr(3, 2), nullptr, 16))};
    }
    std::string out;
    for (size_t i = 0; i < piece.size();) {
        if (piece.compare(i, 3, "\xE2\x96\x81") == 0) {  // U+2581
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(piece[i++]);
        }
    }
    return raw_bytes(out);
}

ov::OutputVector parse_bbpe_config(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                   ov::OutputVector inputs,
                                   const FactoryCreateType& create_func,
                                   bool byte_encode = true) {
    // 1. Parse vocab and add as input
    std::vector<std::string> vocab_from_config{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        vocab_from_config = *val;
    }
    auto vocab = parse_bbpe_vocab(vocab_from_config, byte_encode);
    auto vocab_const = create_string_constant(vocab);

    inputs.insert(inputs.end(), vocab_const.begin(), vocab_const.end());

    // 2. Parse merges
    std::vector<std::string> merges{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("merges"))) {
        merges = *val;
    }

    std::vector<std::vector<uint8_t>> left_merges;
    std::vector<std::vector<uint8_t>> right_merges;

    auto encode_piece = [&](const std::string& piece) {
        return byte_encode ? apply_unicode_to_bytes(piece) : std::vector<uint8_t>(piece.begin(), piece.end());
    };
    // llama.cpp never applies a merge whose result is not a token; the BPE op looks every merge up and
    // fails on such entries (Surya OCR ships a single placeholder merge "a a").
    const std::unordered_set<std::string> vocab_set(vocab_from_config.begin(), vocab_from_config.end());
    for (const auto& merge : merges) {
        size_t space = merge.find(' ');
        std::string left = merge.substr(0, space);
        std::string right = merge.substr(space + 1);
        if (!vocab_set.count(left) || !vocab_set.count(right) || !vocab_set.count(left + right)) {
            continue;
        }

        left_merges.push_back(encode_piece(left));
        right_merges.push_back(encode_piece(right));
    }

    auto left_merges_const = create_string_constant(left_merges);
    auto right_merges_const = create_string_constant(right_merges);

    inputs.insert(inputs.end(), left_merges_const.begin(), left_merges_const.end());
    inputs.insert(inputs.end(), right_merges_const.begin(), right_merges_const.end());

    // 3. Extract special tokens
    ov::Tensor token_types{};
    std::vector<std::string> tokens{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    std::vector<std::vector<uint8_t>> special_tokens;
    std::vector<int32_t> special_token_indices;

    for (size_t i = 0; i < vocab.size(); ++i) {
        if (is_special_token(token_types.data<int32_t>()[i])) {
            special_tokens.push_back(vocab[i]);
            special_token_indices.push_back(static_cast<int32_t>(i));
        }
    }

    auto const_special_tokens = create_string_constant(special_tokens);
    inputs.insert(inputs.end(), const_special_tokens.begin(), const_special_tokens.end());
    ov::Output<ov::Node> const_special_token_indices =
        std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_indices.size()}, special_token_indices);
    inputs.push_back(const_special_token_indices);

    // 4. Build BPETokenizer node
    std::string unk_token = "";
    if (auto it = tokenizer_config.find("unknown_token_id");
        it != tokenizer_config.end() && std::holds_alternative<ov::Tensor>(it->second)) {
        const auto& tensor = std::get<ov::Tensor>(it->second);
        uint32_t unknown_token_id = tensor.data<uint32_t>()[0];
        unk_token = vocab_from_config[unknown_token_id];
    }

    int32_t cache_capacity =
        std::max<int32_t>(static_cast<int32_t>(vocab.size() * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY);

    std::map<std::string, ov::Any> attributes = {
        {"unk_token", unk_token},
        {"fuse_unk", true},
        {"suffix_indicator", std::string("")},
        {"end_suffix", std::string("")},
        {"byte_fallback", true},
        {"cache_capacity", cache_capacity},
    };

    return create_func("BPETokenizer", inputs, attributes);
}

// ---- SentencePiece ("llama" tokenizer model) --------------------------------------------------
// The GGUF holds the vocabulary as pieces, scores and token types, which is all a SentencePiece model
// is. It is serialized as a ModelProto (minimal protobuf writer below) and handed to the
// SentencepieceTokenizer / SentencepieceDetokenizer ops of openvino_tokenizers, wired like its converter.
namespace {

void pb_varint(std::string& s, uint64_t v) {
    while (v >= 0x80) {
        s.push_back(static_cast<char>(v | 0x80));
        v >>= 7;
    }
    s.push_back(static_cast<char>(v));
}
void pb_tag(std::string& s, uint32_t field, uint32_t wire) {
    pb_varint(s, (static_cast<uint64_t>(field) << 3) | wire);
}
void pb_int(std::string& s, uint32_t field, int64_t v) {  // negative values take the full 10 bytes
    pb_tag(s, field, 0);
    pb_varint(s, static_cast<uint64_t>(v));
}
void pb_float(std::string& s, uint32_t field, float f) {
    pb_tag(s, field, 5);
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    for (int i = 0; i < 4; i++) s.push_back(static_cast<char>((u >> (8 * i)) & 0xFF));
}
void pb_str(std::string& s, uint32_t field, const std::string& v) {
    pb_tag(s, field, 2);
    pb_varint(s, v.size());
    s += v;
}

// sentencepiece_model.proto: ModelProto{pieces=1, trainer_spec=2, normalizer_spec=3},
// SentencePiece{piece=1, score=2, type=3}, TrainerSpec{model_type=3, vocab_size=4, byte_fallback=35,
// unk_id=40, bos_id=41, eos_id=42, pad_id=43, unk_surface=44},
// NormalizerSpec{name=1, add_dummy_prefix=3, remove_extra_whitespaces=4, escape_whitespaces=5}.
std::string build_sentencepiece_model(const std::vector<std::string>& tokens,
                                      const std::vector<float>& scores,
                                      const std::vector<int32_t>& types,
                                      bool add_space_prefix,
                                      int64_t unk_id,
                                      int64_t bos_id,
                                      int64_t eos_id,
                                      int64_t pad_id,
                                      bool unigram,
                                      bool escape_whitespaces,
                                      bool remove_extra_whitespaces = false) {
    std::string model;
    bool has_bytes = false, has_unk = false;
    for (size_t i = 0; i < tokens.size(); i++) {
        const int32_t t = i < types.size() ? types[i] : 1;
        has_bytes |= t == 6;
        has_unk |= t == 2;
    }
    if (unk_id < 0 || static_cast<size_t>(unk_id) >= tokens.size()) {
        unk_id = 0;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        int32_t t = i < types.size() ? types[i] : 1;
        if (t == 0) t = 1;                                               // undefined -> normal
        if (!has_unk && static_cast<int64_t>(i) == unk_id) t = 2;        // SentencePiece needs an unknown piece
        if (t == 2 && static_cast<int64_t>(i) != unk_id) t = 5;          // only one unknown piece; padding slots are unused
        std::string piece;
        pb_str(piece, 1, tokens[i]);
        pb_float(piece, 2, i < scores.size() ? scores[i] : 0.0f);
        pb_int(piece, 3, t);
        pb_str(model, 1, piece);
    }
    std::string trainer;
    // BPE merges the adjacent pair whose concatenation scores highest, as llama.cpp's SPM does; PLaMo-2
    // instead picks the segmentation with the best total score, which is a unigram model.
    pb_int(trainer, 3, unigram ? 1 : 2);
    pb_int(trainer, 4, static_cast<int64_t>(tokens.size()));
    pb_int(trainer, 35, has_bytes ? 1 : 0);
    pb_int(trainer, 40, unk_id);
    pb_int(trainer, 41, bos_id);
    pb_int(trainer, 42, eos_id);
    pb_int(trainer, 43, pad_id);
    pb_str(trainer, 44, tokens[static_cast<size_t>(unk_id)]);
    pb_str(model, 2, trainer);
    std::string normalizer;
    pb_str(normalizer, 1, "identity");
    pb_int(normalizer, 3, add_space_prefix ? 1 : 0);
    pb_int(normalizer, 4, remove_extra_whitespaces ? 1 : 0);
    pb_int(normalizer, 5, escape_whitespaces ? 1 : 0);
    pb_str(model, 3, normalizer);
    return model;
}

}  // namespace

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>> create_sentencepiece_tokenizer(
    const std::map<std::string, GGUFMetaData>& tokenizer_config,
    const FactoryCreateType& create_func,
    const std::string& model_name = "llama") {
    const bool plamo2 = model_name == "plamo2";
    std::vector<std::string> tokens;
    if (auto v = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) tokens = *v;
    auto tensor_vec = [&](const char* key, auto zero) {
        using T = decltype(zero);
        std::vector<T> out;
        auto it = tokenizer_config.find(key);
        if (it != tokenizer_config.end()) {
            if (auto t = std::get_if<ov::Tensor>(&it->second)) {
                const T* d = t->data<T>();
                out.assign(d, d + t->get_size());
            }
        }
        return out;
    };
    auto id_of = [&](const char* key, int64_t fallback) {
        auto it = tokenizer_config.find(key);
        if (it != tokenizer_config.end()) {
            if (auto t = std::get_if<ov::Tensor>(&it->second)) return static_cast<int64_t>(t->data<uint32_t>()[0]);
        }
        return fallback;
    };
    const auto scores = tensor_vec("scores", 0.0f);
    auto types = tensor_vec("token_type", int32_t{0});
    // llama.cpp treats these end-of-generation markers as control tokens even when the GGUF types them
    // as normal text (older Gemma files do), so they are matched in the prompt and dropped on decode.
    static const std::set<std::string> eog_markers = {
        "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
        "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
        "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~["};
    types.resize(tokens.size(), 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        if (types[i] != 3 && types[i] != 6 && eog_markers.count(tokens[i])) types[i] = 3;
    }
    // SentencePiece byte fallback needs exactly the 256 pieces "<0x00>".."<0xFF>". Some GGUFs type other
    // tokens as bytes too (Xverse has "<b'\x00'>"); those are ordinary pieces here, and without a
    // complete set the fallback is off altogether.
    auto is_hex_byte = [](const std::string& t) {
        auto hex = [](char c) { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F'); };
        return t.size() == 6 && t.compare(0, 3, "<0x") == 0 && hex(t[3]) && hex(t[4]) && t[5] == '>';
    };
    size_t n_bytes = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (types[i] == 6 && !is_hex_byte(tokens[i])) types[i] = 1;
        n_bytes += types[i] == 6;
    }
    if (n_bytes != 256) {
        for (auto& t : types) {
            if (t == 6) t = 1;
        }
    }
    bool add_space_prefix = true;
    if (auto it = tokenizer_config.find("add_space_prefix"); it != tokenizer_config.end()) {
        if (auto t = std::get_if<ov::Tensor>(&it->second)) add_space_prefix = t->get_byte_size() > 0 && t->data<uint8_t>()[0] != 0;
    }
    // llama.cpp's Unigram normalizer starts every run of non-blanks with a "▁" when it also merges blanks,
    // which is a dummy prefix plus whitespace collapsing, whatever add_space_prefix says.
    bool remove_extra_ws = false;
    if (auto it = tokenizer_config.find("remove_extra_whitespaces"); it != tokenizer_config.end() && model_name == "t5") {
        if (auto t = std::get_if<ov::Tensor>(&it->second)) remove_extra_ws = t->get_byte_size() > 0 && t->data<uint8_t>()[0] != 0;
    }
    add_space_prefix = add_space_prefix || remove_extra_ws;
    const int64_t unk = id_of("unknown_token_id", 0), bos = id_of("bos_token_id", -1), eos = id_of("eos_token_id", -1),
                  pad = id_of("padding_token_id", -1);
    const std::string proto = build_sentencepiece_model(tokens, scores, types, add_space_prefix, unk, bos, eos, pad,
                                                        /*unigram=*/plamo2 || model_name == "t5", /*escape_whitespaces=*/!plamo2,
                                                        remove_extra_ws);
    auto sp_model = std::make_shared<v0::Constant>(element::u8, ov::Shape{proto.size()}, proto.data())->output(0);

    // ---- tokenizer ----
    auto input = std::make_shared<v0::Parameter>(element::string, PartialShape{Dimension::dynamic()});
    // Control and user-defined pieces found in the text are single tokens (llama.cpp parses them as special).
    std::vector<std::pair<std::string, int32_t>> specials;
    for (size_t i = 0; i < tokens.size() && i < types.size(); i++) {
        if (is_special_token(types[i]) && !tokens[i].empty()) specials.emplace_back(tokens[i], static_cast<int32_t>(i));
    }
    std::sort(specials.begin(), specials.end(), std::greater<>());
    std::vector<std::string> special_strings;
    std::vector<int32_t> special_ids;
    for (const auto& sp : specials) {
        special_strings.push_back(sp.first);
        special_ids.push_back(sp.second);
    }
    ov::OutputVector inputs = {sp_model, input->output(0)};
    if (!specials.empty()) {
        auto strs = create_string_constant(special_strings);
        inputs.insert(inputs.end(), strs.begin(), strs.end());
        inputs.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{special_ids.size()}, special_ids)->output(0));
    }
    // GGUF models get no BOS/EOS from the tokenizer itself (the chat template, or the caller, supplies them).
    auto sp = create_func("SentencepieceTokenizer",
                          inputs,
                          {{"add_bos", false}, {"add_eos", false}, {"reverse", true}, {"alpha", 1.0f}, {"nbest_size", 1}});
    ov::Output<ov::Node> indices = sp[0], values = sp[1], dense_shape = sp[2];
    auto pad_value = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, static_cast<int32_t>(std::max<int64_t>(pad, 0)));
    auto zero_value = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0);
    auto one_value = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 1);
    auto padded = std::make_shared<v3::Broadcast>(pad_value, dense_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    auto mask_zero = std::make_shared<v3::Broadcast>(zero_value, dense_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    auto ones = std::make_shared<v3::Broadcast>(one_value, std::make_shared<v3::ShapeOf>(values), ov::op::BroadcastType::BIDIRECTIONAL);
    ov::Output<ov::Node> input_ids = std::make_shared<v3::ScatterNDUpdate>(padded, indices, values);
    ov::Output<ov::Node> attention_mask = std::make_shared<v3::ScatterNDUpdate>(mask_zero, indices, ones);
    // reverse=true tokenizes right-aligned (left padding); put the rows back in reading order
    auto last_axis = std::make_shared<v0::Constant>(element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    input_ids = std::make_shared<v1::Reverse>(input_ids, last_axis, v1::Reverse::Mode::INDEX);
    attention_mask = std::make_shared<v1::Reverse>(attention_mask, last_axis, v1::Reverse::Mode::INDEX);
    input_ids = std::make_shared<v0::Convert>(input_ids, element::i64);
    attention_mask = std::make_shared<v0::Convert>(attention_mask, element::i64);
    input_ids.get_tensor().add_names({"input_ids"});
    attention_mask.get_tensor().add_names({"attention_mask"});
    auto tokenizer = std::make_shared<Model>(ov::OutputVector{input_ids, attention_mask}, ParameterVector{input}, "tokenizer");

    // ---- detokenizer ----
    auto token_ids = std::make_shared<v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    auto ids32 = std::make_shared<v0::Convert>(token_ids, element::i32);
    auto text = create_func("SentencepieceDetokenizer", {sp_model, ids32->output(0)}, {});
    auto valid = create_func("UTF8Validate", text, {{"replace_mode", true}});
    auto packed = create_func("StringTensorPack", valid, {});
    packed[0].get_tensor().add_names({"string_output"});
    auto detokenizer = std::make_shared<Model>(packed, ParameterVector{token_ids}, "detokenizer");
    return {tokenizer, detokenizer};
}

// WordPiece ("bert") vocabularies: BERT's uncased normalization (NFD, lower case, accents and control
// characters dropped, CJK characters isolated), whitespace/punctuation pre-tokenization, then greedy
// longest-match WordPiece with "##" continuations. [CLS]/[SEP] are added by Tokenizer from the metadata.
// The detokenizer puts a space before each word start and glues continuations.
std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>> create_wordpiece_tokenizer(
    const std::map<std::string, GGUFMetaData>& tokenizer_config,
    FactoryCreateType create_func) {
    std::vector<std::string> tokens;
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) tokens = *val;
    ov::Tensor token_types;
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) token_types = *val;
    int32_t unk = 0;
    if (auto it = tokenizer_config.find("unknown_token_id"); it != tokenizer_config.end()) {
        if (auto t = std::get_if<ov::Tensor>(&it->second)) unk = static_cast<int32_t>(t->data<uint32_t>()[0]);
    }
    auto text_const = [](const std::string& text) {
        return std::make_shared<v0::Constant>(element::u8, ov::Shape{text.size()}, text.data())->output(0);
    };
    auto regex_norm = [&](ov::OutputVector in, const std::string& search, const std::string& replace) {
        in.push_back(text_const(search));
        in.push_back(text_const(replace));
        return create_func("RegexNormalization", in, {});
    };

    auto tokenizer_input = std::make_shared<v0::Parameter>(element::string, PartialShape{Dimension::dynamic()});
    OutputVector outputs = create_func("StringTensorUnpack", {tokenizer_input}, {});
    outputs = create_func("CharsMapNormalization", outputs, {{"normalization_form", std::string("nfd")}, {"remove_extra_whitespaces", false}});
    outputs = create_func("CharsMapNormalization", outputs, {{"normalization_form", std::string("identity")}, {"case_fold", true}, {"remove_extra_whitespaces", false}});
    outputs = regex_norm(outputs, "\\p{Mn}", "");                    // strip accents
    outputs = regex_norm(outputs, "[\\p{Cc}&&[^\\s]]", "");        // control characters
    outputs = regex_norm(outputs, "([\\p{Han}])", " $1 ");           // CJK characters stand alone
    outputs = add_ragged_dimension(outputs);

    std::vector<std::string> special_tokens;
    const auto* types = token_types.data<int32_t>();
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special_token(types[i])) special_tokens.push_back(tokens[i]);
    }
    const std::string special_re = join_special_tokens(special_tokens);
    ov::OutputVector split_in = outputs;
    split_in.push_back(text_const(special_re));
    outputs = create_func("SpecialTokensSplit", split_in, {});

    auto split = [&](const std::string& re, const std::string& behaviour) {
        outputs.push_back(text_const(re));
        outputs = create_func("RegexSplit", outputs, {{"behaviour", behaviour}, {"invert", false}, {"max_splits", -1}});
    };
    split("\\s+", "remove");
    split("[!-/]|[:-@]|[\\[-`]|[{-~]|[\\p{P}]|[\\x{4E00}-\\x{9FFF}]|[\\x{3400}-\\x{4DBF}]|[\\x{20000}-\\x{2A6DF}]|"
          "[\\x{2A700}-\\x{2B73F}]|[\\x{2B740}-\\x{2B81F}]|[\\x{2B820}-\\x{2CEAF}]|[\\x{F900}-\\x{FAFF}]|[\\x{2F800}-\\x{2FA1F}]",
          "isolate");

    ov::OutputVector wp_in(outputs.begin(), outputs.begin() + 5);
    // llama.cpp stores BERT vocabularies SentencePiece-style: "▁x" starts a word, a bare "x" continues one.
    // WordPiece wants the original spelling: "x" and "##x".
    const std::string word_start = "\xE2\x96\x81";
    std::vector<std::string> wp_vocab;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& t = tokens[i];
        if (t.compare(0, word_start.size(), word_start) == 0) {
            wp_vocab.push_back(t.substr(word_start.size()));
        } else {
            wp_vocab.push_back(is_special_token(types[i]) ? t : "##" + t);
        }
    }
    auto vocab_const = create_string_constant(wp_vocab);
    wp_in.insert(wp_in.end(), vocab_const.begin(), vocab_const.end());
    wp_in.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, unk)->output(0));
    outputs = create_func("WordpieceTokenizer", wp_in, {{"suffix_indicator", std::string("##")}, {"max_bytes_per_word", 100}});

    // right padding: the sequences are bidirectional, so the side carries no meaning
    ov::Output<ov::Node> ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    auto reduce_axis = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    ov::Output<ov::Node> max_length_batch = std::make_shared<v1::ReduceMax>(ends_minus_begins, reduce_axis, false);
    ov::OutputVector ragged_in = outputs;
    ragged_in.push_back(max_length_batch);
    ragged_in.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0)->output(0));
    outputs = create_func("RaggedToDense", ragged_in, {{"pad_right", true}, {"pad_max_length", false}});
    outputs[0] = std::make_shared<v0::Convert>(outputs[0], element::i64)->output(0);
    outputs[1] = std::make_shared<v0::Convert>(outputs[1], element::i64)->output(0);
    outputs[0].get_tensor().add_names({"input_ids"});
    outputs[1].get_tensor().add_names({"attention_mask"});
    auto tokenizer = std::make_shared<Model>(outputs, ParameterVector{tokenizer_input}, "tokenizer");

    // ---- detokenizer ----
    auto detokenizer_input = std::make_shared<v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});
    std::vector<std::string> pieces;
    for (const auto& t : tokens) {
        pieces.push_back(t.compare(0, word_start.size(), word_start) == 0 ? " " + t.substr(word_start.size()) : t);
    }
    ov::OutputVector detok_in = {detokenizer_input};
    auto pieces_const = create_string_constant(pieces);
    detok_in.insert(detok_in.end(), pieces_const.begin(), pieces_const.end());
    std::vector<int32_t> skip_ids;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special_token(types[i])) skip_ids.push_back(static_cast<int32_t>(i));
    }
    auto skip_const = std::make_shared<v0::Constant>(element::i32, ov::Shape{skip_ids.size()}, skip_ids);
    auto zero = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    auto one = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 1);
    auto int_max = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, std::numeric_limits<int32_t>::max());
    auto skips = std::make_shared<v8::Slice>(skip_const, zero, int_max, one)->outputs();
    detok_in.insert(detok_in.end(), skips.begin(), skips.end());
    auto decoded = create_func("VocabDecoder", detok_in, {});
    ov::OutputVector fuse_in(decoded.begin(), decoded.end() - 1);
    auto fused = create_func("FuzeRagged", fuse_in, {});
    fused.insert(fused.end(), decoded.end() - 1, decoded.end());
    auto valid = create_func("UTF8Validate", fused, {{"replace_mode", true}});
    auto packed = create_func("StringTensorPack", valid, {});
    packed[0].get_tensor().add_names({"string_output"});
    auto detokenizer = std::make_shared<Model>(packed, ParameterVector{detokenizer_input}, "detokenizer");
    return {tokenizer, detokenizer};
}

// Build a minimal SentencePiece ModelProto (raw protobuf, no sentencepiece proto headers) from
// GGUF vocab arrays. Fields used: pieces (1: piece str, 2: score f32, 3: type varint),
// trainer_spec (2: model_type, byte_fallback), normalizer_spec (3: name, add_dummy_prefix).
// GGUF and SP token_type values match: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED,
// 6=BYTE. SP allows only one UNKNOWN piece; extra GGUF-type-2 entries (pad tokens) become
// USER_DEFINED.
static void pb_append_varint(std::vector<uint8_t>& buf, uint64_t value) {
    while (value >= 0x80) {
        buf.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
        value >>= 7;
    }
    buf.push_back(static_cast<uint8_t>(value));
}

static void pb_append_tag_len(std::vector<uint8_t>& buf, uint32_t field, const std::vector<uint8_t>& msg) {
    pb_append_varint(buf, (static_cast<uint64_t>(field) << 3) | 2);
    pb_append_varint(buf, msg.size());
    buf.insert(buf.end(), msg.begin(), msg.end());
}

static std::vector<uint8_t> build_spm_model_proto(const std::vector<std::string>& tokens,
                                                   const std::vector<float>& scores,
                                                   const std::vector<int32_t>& token_types,
                                                   bool add_dummy_prefix,
                                                   bool remove_extra_whitespaces,
                                                   int32_t unk_id = -1,
                                                   int32_t bos_id = -1,
                                                   int32_t eos_id = -1,
                                                   int32_t pad_id = -1) {
    std::vector<uint8_t> proto;
    // SentencePiece resolves BOS/EOS/UNK/PAD by piece string (TrainerSpec.bos_piece etc.), not by
    // id, so override those strings with the actual vocab pieces at the GGUF's token ids.
    auto piece_for = [&](int32_t id) -> std::string {
        return (id >= 0 && static_cast<size_t>(id) < tokens.size()) ? tokens[id] : std::string();
    };
    const std::string unk_piece = piece_for(unk_id);
    const std::string bos_piece = piece_for(bos_id);
    const std::string eos_piece = piece_for(eos_id);
    const std::string pad_piece = piece_for(pad_id);
    bool unk_seen = false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        int32_t gguf_type = token_types[i];
        int32_t sp_type = gguf_type;
        if (gguf_type == 2) {  // UNKNOWN
            if (!unk_seen) {
                unk_seen = true;
            } else {
                sp_type = 4;  // promote extra UNKNOWN → USER_DEFINED (pad tokens)
            }
        }
        // Build SentencePiece piece sub-message
        std::vector<uint8_t> piece_msg;
        // field 1: piece (string), tag = 0x0a
        pb_append_varint(piece_msg, (1ULL << 3) | 2);
        const auto& tok = tokens[i];
        pb_append_varint(piece_msg, tok.size());
        piece_msg.insert(piece_msg.end(), tok.begin(), tok.end());
        // field 2: score (float32), tag = 0x15
        piece_msg.push_back(0x15);
        float f = scores[i];
        uint8_t fb[4];
        std::memcpy(fb, &f, 4);
        piece_msg.insert(piece_msg.end(), fb, fb + 4);
        // field 3: type (varint), tag = 0x18
        piece_msg.push_back(0x18);
        pb_append_varint(piece_msg, static_cast<uint64_t>(sp_type));
        // Append as field 1 (pieces) of ModelProto
        pb_append_tag_len(proto, 1, piece_msg);
    }

    // trainer_spec (field 2): model_type=BPE, byte_fallback=true, special-token ids/pieces from
    // the GGUF (SentencePiece's own bos_id=1/eos_id=2/unk_id=0 defaults don't match GGUF ids).
    {
        std::vector<uint8_t> ts;
        ts.push_back(0x18);  // field 3 varint (model_type)
        pb_append_varint(ts, 2);  // BPE
        auto put_id = [&ts](uint32_t field, int32_t id) {
            if (id < 0)
                return;
            pb_append_varint(ts, (static_cast<uint64_t>(field) << 3) | 0);  // varint wire type
            pb_append_varint(ts, static_cast<uint64_t>(id));
        };
        auto put_str = [&ts](uint32_t field, const std::string& s) {
            if (s.empty())
                return;
            pb_append_varint(ts, (static_cast<uint64_t>(field) << 3) | 2);  // length-delimited
            pb_append_varint(ts, s.size());
            ts.insert(ts.end(), s.begin(), s.end());
        };
        // ids (TrainerSpec 40-43) and piece strings (45-48); bos_id() etc. resolve via the piece.
        put_id(40, unk_id);
        put_id(41, bos_id);
        put_id(42, eos_id);
        put_id(43, pad_id);
        put_str(45, unk_piece);
        put_str(46, bos_piece);
        put_str(47, eos_piece);
        put_str(48, pad_piece);
        pb_append_varint(ts, (35ULL << 3) | 0);  // field 35 varint (byte_fallback)
        ts.push_back(1);          // byte_fallback = true
        pb_append_tag_len(proto, 2, ts);
    }

    // normalizer_spec (field 3): name="identity", add_dummy_prefix (leading metaspace) from GGUF
    // tokenizer.ggml.add_space_prefix (gemma3 sets it false).
    {
        std::vector<uint8_t> ns;
        // field 1: name = "identity"
        std::string name = "identity";
        pb_append_varint(ns, (1ULL << 3) | 2);
        pb_append_varint(ns, name.size());
        ns.insert(ns.end(), name.begin(), name.end());
        // field 3: add_dummy_prefix
        pb_append_varint(ns, (3ULL << 3) | 0);
        ns.push_back(add_dummy_prefix ? 1 : 0);
        pb_append_varint(ns, (4ULL << 3) | 0);
        ns.push_back(remove_extra_whitespaces ? 1 : 0);
        pb_append_tag_len(proto, 3, ns);
    }

    return proto;
}

// Convert SentencepieceTokenizer's sparse output (indices [N,2], values [N], dense_shape [2])
// into ragged begins/ends used by the rest of the tokenizer pipeline.
// row_splits[B+1]: row_splits[i] = number of tokens for rows 0..i-1 (cumulative).
// begins[i] = row_splits[i], ends[i] = row_splits[i+1].
static ov::OutputVector sparse_to_ragged(const ov::Output<ov::Node>& sparse_indices,
                                          const ov::Output<ov::Node>& sparse_values,
                                          const ov::Output<ov::Node>& dense_shape) {
    // 1-D i64 0 and 1, used both as axes and as slice bounds
    auto k0_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto k1_1d = std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto ax0_0d = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
    auto one_i64 = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});
    auto zero_i64 = std::make_shared<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});

    // batch_indices = sparse_indices[:, 0]
    auto batch_indices = std::make_shared<v8::Gather>(sparse_indices, ax0_0d, k1_1d);
    // B = dense_shape[0] (batch size)
    auto B = std::make_shared<v8::Gather>(dense_shape, ax0_0d, ax0_0d);
    auto B_plus_1 = std::make_shared<v1::Add>(B, one_i64);
    // range [0, B+1) as row boundaries
    auto range = std::make_shared<v4::Range>(zero_i64, B_plus_1, one_i64, element::i64);
    // mask[n, i] = batch_indices[n] < range[i]  → row_splits[i] = sum of mask[:, i]
    auto bi_unsq = std::make_shared<v0::Unsqueeze>(batch_indices, k1_1d);
    auto range_unsq = std::make_shared<v0::Unsqueeze>(range, k0_1d);
    auto mask = std::make_shared<v1::Less>(bi_unsq, range_unsq);
    auto mask_i32 = std::make_shared<v0::Convert>(mask, element::i32);
    auto row_splits = std::make_shared<v1::ReduceSum>(mask_i32, k0_1d, false)->output(0);

    // begins = row_splits[0:B], ends = row_splits[1:B+1]
    auto int64_max = std::make_shared<v0::Constant>(element::i64, Shape{1},
                                                     std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    // B as a 1D i64 tensor (v8::Slice stop must be 1D)
    auto B_i64_1d = std::make_shared<v0::Unsqueeze>(B, k0_1d);
    auto begins = std::make_shared<v8::Slice>(row_splits, k0_1d, B_i64_1d, k1_1d, k0_1d)->output(0);
    auto ends_node = std::make_shared<v8::Slice>(row_splits, k1_1d, int64_max, k1_1d, k0_1d)->output(0);

    auto values_i32 = std::make_shared<v0::Convert>(sparse_values, element::i32)->output(0);
    return {begins, ends_node, values_i32};
}

// Scalar bool/id lookups over the GGUF tokenizer metadata. Values are typically a scalar Tensor;
// a plain int is also accepted.
static bool read_tokenizer_flag(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                const char* key,
                                bool dflt) {
    if (auto it = tokenizer_config.find(key); it != tokenizer_config.end()) {
        if (std::holds_alternative<ov::Tensor>(it->second)) {
            const auto& t = std::get<ov::Tensor>(it->second);
            if (t.get_size() > 0)
                return t.data<bool>()[0];
        } else if (std::holds_alternative<int>(it->second)) {
            return std::get<int>(it->second) != 0;
        }
    }
    return dflt;
}

static int32_t read_tokenizer_id(const std::map<std::string, GGUFMetaData>& tokenizer_config, const char* key) {
    if (auto it = tokenizer_config.find(key); it != tokenizer_config.end()) {
        if (std::holds_alternative<ov::Tensor>(it->second)) {
            const auto& t = std::get<ov::Tensor>(it->second);
            if (t.get_size() > 0)
                return static_cast<int32_t>(t.data<uint32_t>()[0]);
        } else if (std::holds_alternative<int>(it->second)) {
            return static_cast<int32_t>(std::get<int>(it->second));
        }
    }
    return -1;
}

// SentencePiece requires exactly one UNKNOWN token (type=2). Some models (gemma2, gemma3) have
// no type-2 token in their GGUF vocab but do supply unknown_token_id. Promote it.
static void promote_unknown_token_type(std::vector<int32_t>& token_types,
                                       const std::map<std::string, GGUFMetaData>& tokenizer_config) {
    if (std::any_of(token_types.begin(), token_types.end(), [](int32_t t) { return t == 2; }))
        return;
    const int32_t unk_id = read_tokenizer_id(tokenizer_config, "unknown_token_id");
    if (unk_id >= 0 && static_cast<size_t>(unk_id) < token_types.size())
        token_types[unk_id] = 2;
}

// A u8 Constant holding the bytes of @p str, as expected by the string ops.
static ov::Output<ov::Node> u8_string_constant(const std::string& str) {
    ov::Tensor tensor(ov::element::u8, {str.size()});
    std::memcpy(tensor.data<uint8_t>(), str.data(), str.size());
    return std::make_shared<v0::Constant>(tensor)->output(0);
}

// Build SentencePiece (model="llama") tokenizer outputs from a serialized ModelProto via the
// SentencepieceTokenizer op.
static ov::OutputVector parse_spm_config(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                          ov::OutputVector inputs,
                                          const FactoryCreateType& create_func) {
    // tokenizer_config outlives this call, so read the vocabulary in place instead of copying it.
    const auto* vocab_ptr = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"));
    ov::Tensor token_types_tensor;
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type")))
        token_types_tensor = *val;

    ov::Tensor scores_tensor;
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("scores")))
        scores_tensor = *val;

    OPENVINO_ASSERT(vocab_ptr && !vocab_ptr->empty(),
                    "[gguf tokenizer] SentencePiece: 'tokens' array is missing or empty");
    const auto& vocab = *vocab_ptr;
    OPENVINO_ASSERT(scores_tensor.get_size() == vocab.size(),
                    "[gguf tokenizer] SentencePiece: 'scores' tensor size (", scores_tensor.get_size(),
                    ") != vocab size (", vocab.size(), ")");
    OPENVINO_ASSERT(token_types_tensor.get_size() == vocab.size(),
                    "[gguf tokenizer] SentencePiece: 'token_type' array size != vocab size");

    std::vector<float> scores(scores_tensor.data<float>(),
                              scores_tensor.data<float>() + scores_tensor.get_size());
    std::vector<int32_t> token_types(token_types_tensor.data<int32_t>(),
                                     token_types_tensor.data<int32_t>() + token_types_tensor.get_size());

    promote_unknown_token_type(token_types, tokenizer_config);

    // Leading metaspace prefix; gemma3 sets it false.
    const bool add_space_prefix = read_tokenizer_flag(tokenizer_config, "add_space_prefix", true);

    // Special-token ids from the GGUF, baked into the SP proto's trainer_spec.
    const int32_t unk_id = read_tokenizer_id(tokenizer_config, "unknown_token_id");
    const int32_t bos_id = read_tokenizer_id(tokenizer_config, "bos_token_id");
    const int32_t eos_id = read_tokenizer_id(tokenizer_config, "eos_token_id");
    const int32_t pad_id = read_tokenizer_id(tokenizer_config, "padding_token_id");

    // Build the serialized ModelProto and wrap as a u8 Constant (first input to SentencepieceTokenizer)
    auto proto_bytes =
        build_spm_model_proto(vocab, scores, token_types, add_space_prefix, true, unk_id, bos_id, eos_id, pad_id);
    auto sp_model_const = std::make_shared<v0::Constant>(element::u8, Shape{proto_bytes.size()}, proto_bytes.data());

    // inputs = SpecialTokensSplit outputs: [ragged_begins(0), ragged_ends(1), begins(2), ends(3), chars(4), ...]
    // SentencepieceTokenizer with 4 inputs: (sp_model, begins, ends, chars) — the inner flat ragged string.
    OPENVINO_ASSERT(inputs.size() >= 5, "[gguf tokenizer] SentencePiece: expected >=5 outputs from SpecialTokensSplit");
    ov::OutputVector sp_inputs = {sp_model_const->output(0), inputs[2], inputs[3], inputs[4]};
    // SentencePiece deliberately does not encode CONTROL pieces from their spelling.
    // Chat delimiters and media markers must use their GGUF IDs even when automatic
    // BOS/EOS insertion is disabled. Use the tokenizer op's explicit special-token map.
    std::vector<std::vector<uint8_t>> special_pieces;
    std::vector<int32_t> special_ids;
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (is_special_token(token_types[i])) {
            special_pieces.emplace_back(vocab[i].begin(), vocab[i].end());
            special_ids.push_back(static_cast<int32_t>(i));
        }
    }
    if (!special_ids.empty()) {
        auto pieces = create_string_constant(special_pieces);
        sp_inputs.insert(sp_inputs.end(), pieces.begin(), pieces.end());
        sp_inputs.push_back(v0::Constant::create(element::i32, Shape{special_ids.size()}, special_ids));
    }
    // BOS/EOS are not baked into SentencepieceTokenizer: its add_bos/add_eos are compile-time
    // attributes that add_special_tokens=false can't switch off. Emit them as a CombineSegments
    // segment instead, like the BPE paths (see the CombineSegments block below).
    auto sp_tok = create_func("SentencepieceTokenizer", sp_inputs,
                               {{"nbest_size", int32_t{0}},
                                {"alpha", 0.0f},
                                {"add_bos", false},
                                {"add_eos", false},
                                {"reverse", false}});
    // sp_tok = [sparse_indices [N,2] i64, sparse_values [N] i32, dense_shape [2] i64]
    OPENVINO_ASSERT(sp_tok.size() == 3, "[gguf tokenizer] SentencepieceTokenizer must have 3 outputs");
    // SP tokenizes each inner segment independently; map its sparse output to an inner ragged
    // form, then FuzeRagged collapses back to batch-level using the outer ragged begins/ends.
    auto inner = sparse_to_ragged(sp_tok[0], sp_tok[1], sp_tok[2]);
    // inner = [begins(N_inner), ends(N_inner), values]
    auto fused = create_func("FuzeRagged", {inputs[0], inputs[1], inner[0], inner[1]}, {});
    // fused = [batch_begins(batch), batch_ends(batch)]
    return {fused[0], fused[1], inner[2]};
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
build_tokenizer_models(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                       std::map<std::string, GGUFMetaData> tokenizer_config) {
    auto tokenizer_input = std::make_shared<v0::Parameter>(element::string, PartialShape{Dimension::dynamic()});

    FactoryCreateType create_func =
        reinterpret_cast<FactoryCreateType>(get_symbol(shared_object_ov_tokenizers, "create_tokenizer_node"));

    std::string model{};
    if (auto val = std::get_if<std::string>(&tokenizer_config.at("model"))) {
        model = *val;
    }
    // Unigram SentencePiece and WordPiece build their own graphs; BOS/EOS come from TokenizerImpl::add_gguf_special_tokens.
    if (model == "t5" || model == "plamo2") {
        auto [tok, detok] = create_sentencepiece_tokenizer(tokenizer_config, create_func, model);
        return {tok, detok, tokenizer_config};
    }
    if (model == "bert") {
        auto [tok, detok] = create_wordpiece_tokenizer(tokenizer_config, create_func);
        return {tok, detok, tokenizer_config};
    }
    // plamo2 uses the same SentencePiece BPE format as llama
    const std::string effective_model = (model == "plamo2") ? "llama" : model;
    OPENVINO_ASSERT(effective_model == "gpt2" || effective_model == "llama" || effective_model == "gemma4",
                    "[gguf tokenizer] Unsupported tokenizer model '", model,
                    "'. Supported: 'gpt2' (BPE), 'llama'/'plamo2' (SentencePiece BPE), 'gemma4' (SPM-style BPE).");

    OutputVector outputs = create_func("StringTensorUnpack", {tokenizer_input}, {});

    // gemma4 is an SPM-style BPE: escape whitespace to U+2581 (metaspace) on the raw decomposed
    // string (indices 0,1,2 = begins/ends/chars) before the ragged dimension is added. It does
    // NOT prepend a leading metaspace (llama.cpp gemma4: add_space_prefix=false), so only the
    // space->metaspace replacement is applied.
    if (effective_model == "gemma4") {
        const std::string metaspace = "\xe2\x96\x81";  // U+2581 ▁
        ov::OutputVector in(outputs.begin(), outputs.begin() + 3);
        in.push_back(u8_string_constant(" "));
        in.push_back(u8_string_constant(metaspace));
        auto normed = create_func("RegexNormalization", in, {{"global_replace", true}});
        for (size_t i = 0; i < normed.size() && i < outputs.size(); ++i) {
            outputs[i] = normed[i];
        }
    }

    outputs = add_ragged_dimension(outputs);

    // Special token filtering
    std::vector<std::string> tokens, special_tokens;
    ov::Tensor token_types;
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    const auto token_types_data = token_types.data<int32_t>();

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special_token(token_types_data[i])) {
            special_tokens.push_back(tokens[i]);
        }
    }

    std::string special_tokens_re = join_special_tokens(special_tokens);

    ov::Tensor ov_special_tokens(ov::element::u8, {special_tokens_re.size()});
    std::memcpy(ov_special_tokens.data<uint8_t>(), special_tokens_re.data(), special_tokens_re.size());
    auto const_special_tokens = std::make_shared<v0::Constant>(ov_special_tokens);

    ov::OutputVector inputs_to_split = outputs;
    inputs_to_split.push_back(const_special_tokens->output(0));
    outputs = create_func("SpecialTokensSplit", inputs_to_split, {});

    if (effective_model == "llama") {
        // SentencePiece: SP handles word splitting internally; skip BPE-style RegexSplit.
        // outputs[0..2] = ragged string (begins, ends, chars) from SpecialTokensSplit.
        outputs = parse_spm_config(tokenizer_config, outputs, create_func);
        // outputs = [begins, ends, values] (ragged token ids, i32)
    } else {
        // BPE: gpt2 is byte-level BPE with regex pre-tokenization; gemma4 is SPM-style BPE over
        // the raw-UTF8 vocab, split only on newlines (mirrors llama.cpp's GEMMA4 pre-type).
        if (effective_model == "gemma4") {
            const std::string newline_split = "[^\\n]+|[\\n]+";
            ov::Tensor ov_split_re(ov::element::u8, {newline_split.size()});
            std::memcpy(ov_split_re.data<uint8_t>(), newline_split.data(), newline_split.size());
            auto const_ov_split_re = std::make_shared<v0::Constant>(ov_split_re);
            outputs.push_back(const_ov_split_re->output(0));
            outputs = create_func("RegexSplit", outputs,
                                   {{"behaviour", std::string("isolate")}, {"invert", false}, {"max_splits", -1}});

            ov::OutputVector bbpe_inputs(outputs.begin(), outputs.begin() + 5);
            outputs = parse_bbpe_config(tokenizer_config, bbpe_inputs, create_func, /*byte_encode=*/false);
        } else {
            // BPE (gpt2): apply regex pre-tokenization splits before the BPE encoder.
            std::string pre{};
            if (auto val = std::get_if<std::string>(&tokenizer_config.at("pre"))) {
                pre = *val;
            }
            for (const auto& split_re : get_split_regex(pre)) {
                ov::Tensor ov_split_re(ov::element::u8, {split_re.size()});
                std::memcpy(ov_split_re.data<uint8_t>(), split_re.data(), split_re.size());
                auto const_ov_split_re = std::make_shared<v0::Constant>(ov_split_re);
                outputs.push_back(const_ov_split_re->output(0));
                outputs = create_func("RegexSplit", outputs,
                                      {{"behaviour", std::string("isolate")}, {"invert", false}, {"max_splits", -1}});
            }
            ov::OutputVector bbpe_inputs(outputs.begin(), outputs.begin() + 5);
            outputs = parse_bbpe_config(tokenizer_config, bbpe_inputs, create_func);
        }
    }

    ov::Output<ov::Node> max_length = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, MAX_LENGTH);
    ov::Output<ov::Node> ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    max_length = std::make_shared<v1::Minimum>(ends_minus_begins, max_length);
    outputs[0] = std::make_shared<v1::Subtract>(outputs[1], max_length)->output(0);

    // BOS/EOS as an explicit CombineSegments segment, for every tokenizer flavour: BPE has no
    // add_bos mechanism at all, and SentencepieceTokenizer's add_bos/add_eos are compile-time
    // attributes, so neither backend can honor encode(add_special_tokens=false) on its own.
    //
    // Must come after the truncation above: MakeAddSpecialTokensSatateful identifies the
    // main-sequence segment by its begins being a Subtract, and toggles every other segment's
    // `ends` at runtime; emitting this before truncation would break that detection.
    {
        // Defaults follow llama.cpp (llama-vocab.cpp): SPM defaults add_bos true; for BPE the base
        // default is false, but a few pre-tokenizer families flip it to true and gemma4 is forced
        // true. An explicit tokenizer.ggml.add_bos_token in the GGUF always wins over the default.
        std::string pre{};
        if (auto it = tokenizer_config.find("pre"); it != tokenizer_config.end()) {
            if (auto val = std::get_if<std::string>(&it->second)) {
                pre = *val;
            }
        }
        const bool default_add_bos =
            (effective_model == "llama") || (effective_model == "gemma4") || pre_tokenizer_adds_bos(pre);

        const bool add_bos = read_tokenizer_flag(tokenizer_config, "add_bos_token", default_add_bos);
        const bool add_eos = read_tokenizer_flag(tokenizer_config, "add_eos_token", false);
        const int32_t bos_id = read_tokenizer_id(tokenizer_config, "bos_token_id");
        const int32_t eos_id = read_tokenizer_id(tokenizer_config, "eos_token_id");

        ov::OutputVector cs_inputs;
        std::vector<int32_t> segment_ids;
        // One added-token segment is [begins=0, ends=1, ids=[tok]]; `ends` is what the stateful
        // pass rewrites to 0 when the caller asks for add_special_tokens=false.
        auto add_token_segment = [&](int32_t token_id) {
            cs_inputs.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0)->output(0));
            cs_inputs.push_back(std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 1)->output(0));
            cs_inputs.push_back(
                std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, std::vector<int32_t>{token_id})
                    ->output(0));
            segment_ids.push_back(0);
        };

        if (add_bos && bos_id >= 0) {
            add_token_segment(bos_id);
        }
        cs_inputs.push_back(outputs[0]);
        cs_inputs.push_back(outputs[1]);
        cs_inputs.push_back(outputs[2]);
        segment_ids.push_back(0);
        if (add_eos && eos_id >= 0) {
            add_token_segment(eos_id);
        }

        if (segment_ids.size() > 1) {
            cs_inputs.push_back(
                std::make_shared<v0::Constant>(element::i32, ov::Shape{segment_ids.size()}, segment_ids)->output(0));
            auto combined = create_func("CombineSegments", cs_inputs, {});
            // CombineSegments returns [begins, ends, values, ...]; keep the ragged token ids.
            OPENVINO_ASSERT(combined.size() >= 3, "[gguf tokenizer] CombineSegments must have >=3 outputs");
            outputs[0] = combined[0];
            outputs[1] = combined[1];
            outputs[2] = combined[2];
        }
    }

    // Left padding
    ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    auto reduce_axis = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    ov::Output<ov::Node> max_length_batch = std::make_shared<v1::ReduceMax>(ends_minus_begins, reduce_axis, false);

    ov::OutputVector inputs_for_ragged_to_dense = outputs;
    inputs_for_ragged_to_dense.push_back(max_length_batch);
    ov::Output<ov::Node> const_zero_for_rg = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0);
    inputs_for_ragged_to_dense.push_back(const_zero_for_rg);

    outputs =
        create_func("RaggedToDense", inputs_for_ragged_to_dense, {{"pad_right", false}, {"pad_max_length", false}});

    // Convert output types
    outputs[0] = std::make_shared<v0::Convert>(outputs[0], element::i64)->output(0);
    outputs[1] = std::make_shared<v0::Convert>(outputs[1], element::i64)->output(0);
    outputs[0].get_tensor().add_names({"input_ids"});
    outputs[1].get_tensor().add_names({"attention_mask"});

    auto tokenizer = std::make_shared<Model>(outputs, ParameterVector{tokenizer_input}, "tokenizer");

    // DETOKENIZER model
    auto detokenizer_input =
        std::make_shared<v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});

    std::shared_ptr<Model> detokenizer;
    if (effective_model == "llama") {
        // SentencepieceDetokenizer: takes (sp_model_u8, token_ids_i32) → ragged string
        std::vector<int32_t> spm_types(token_types.data<int32_t>(),
                                       token_types.data<int32_t>() + token_types.get_size());
        promote_unknown_token_type(spm_types, tokenizer_config);
        std::vector<float> spm_scores;
        if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("scores"))) {
            const ov::Tensor& st = *val;
            spm_scores.assign(st.data<float>(), st.data<float>() + st.get_size());
        }
        // Mirror the tokenizer's add_space_prefix so detokenization strips the leading
        // metaspace symmetrically; SentencePiece otherwise strips leading whitespace even
        // when add_dummy_prefix is false.
        const bool detok_add_space_prefix = read_tokenizer_flag(tokenizer_config, "add_space_prefix", true);
        auto proto_bytes = build_spm_model_proto(tokens, spm_scores, spm_types, detok_add_space_prefix, false);
        auto sp_model_const =
            std::make_shared<v0::Constant>(element::u8, Shape{proto_bytes.size()}, proto_bytes.data());
        auto ids_i32 = std::make_shared<v0::Convert>(detokenizer_input, element::i32)->output(0);
        auto sp_detok = create_func("SentencepieceDetokenizer",
                                    {sp_model_const->output(0), ids_i32},
                                    {});
        // sp_detok outputs = ragged string (begins, ends, chars) → pack → string tensor
        auto packed = create_func("StringTensorPack", sp_detok, {});
        packed[0].get_tensor().add_names({"string_output"});
        detokenizer = std::make_shared<Model>(packed, ParameterVector{detokenizer_input}, "detokenizer");
    } else {
        // BPE detokenizer: VocabDecoder + FuzeRagged + UTF8Validate. gpt2 uses GPT-2 byte
        // decoding of the vocab; gemma4 pieces decode to raw UTF-8 ("▁" -> space, "<0xHH>" -> the byte).
        std::vector<std::vector<uint8_t>> vocab;
        if (effective_model == "gemma4") {
            for (const auto& t : tokens) vocab.push_back(decoded_piece(t));
        } else {
            vocab = parse_bbpe_vocab(tokens);
        }
        ov::OutputVector const_vocab = create_string_constant(vocab);
        OutputVector detokenizer_outputs = {detokenizer_input};
        detokenizer_outputs.insert(detokenizer_outputs.end(), const_vocab.begin(), const_vocab.end());

        std::vector<int32_t> special_token_ids;
        for (size_t i = 0; i < token_types.get_size(); ++i) {
            if (is_special_token(token_types.data<int32_t>()[i]))
                special_token_ids.push_back(static_cast<int32_t>(i));
        }

        auto special_ids_const =
            std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_ids.size()}, special_token_ids);
        auto const_zero = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
        auto const_one = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 1);
        int32_t int32_max_value = std::numeric_limits<int32_t>::max();
        auto const_int32_max = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, int32_max_value);
        auto sliced_skips =
            std::make_shared<v8::Slice>(special_ids_const, const_zero, const_int32_max, const_one)->outputs();
        detokenizer_outputs.insert(detokenizer_outputs.end(), sliced_skips.begin(), sliced_skips.end());

        detokenizer_outputs = create_func("VocabDecoder", detokenizer_outputs, {});
        ov::OutputVector inputs_for_fused_ragged(detokenizer_outputs.begin(), detokenizer_outputs.end() - 1);
        auto outputs_fused_ragged = create_func("FuzeRagged", inputs_for_fused_ragged, {});
        outputs_fused_ragged.insert(outputs_fused_ragged.end(), detokenizer_outputs.end() - 1, detokenizer_outputs.end());
        ov::OutputVector inputs_for_utf8_validate(outputs_fused_ragged.begin(), outputs_fused_ragged.end());
        auto outputs_utf8_validate =
            create_func("UTF8Validate", inputs_for_utf8_validate, {{"replace_mode", true}});
        auto packed_output = create_func("StringTensorPack", outputs_utf8_validate, {});
        packed_output[0].get_tensor().add_names({"string_output"});
        detokenizer = std::make_shared<Model>(packed_output, ParameterVector{detokenizer_input}, "detokenizer");
    }

    return {tokenizer, detokenizer, tokenizer_config};
}

// Convert the frontend's rt_info tokenizer metadata into the GGUFMetaData config map.
std::map<std::string, GGUFMetaData> tokenizer_config_from_rt_info(const ov::AnyMap& rt_cfg) {
    std::map<std::string, GGUFMetaData> cfg;
    for (const auto& [key, value] : rt_cfg) {
        if (value.is<std::string>()) {
            cfg[key] = value.as<std::string>();
        } else if (value.is<std::vector<std::string>>()) {
            cfg[key] = value.as<std::vector<std::string>>();
        } else if (value.is<ov::Tensor>()) {
            cfg[key] = value.as<ov::Tensor>();
        } else if (value.is<int>()) {
            cfg[key] = value.as<int>();
        } else if (value.is<float>()) {
            cfg[key] = value.as<float>();
        }
        // other types are not part of the tokenizer config and are ignored
    }
    return cfg;
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_config(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                             const std::filesystem::path& gguf_model_path) {
#ifdef ENABLE_GGUF
    auto gguf_metadata = get_gguf_metadata(gguf_model_path.string());
    auto tokenizer_config = tokenizer_config_from_meta(gguf_metadata);
    return build_tokenizer_models(shared_object_ov_tokenizers, std::move(tokenizer_config));
#else
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_parameters(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                                 const ov::AnyMap& tokenizer_metadata) {
    OPENVINO_ASSERT(!tokenizer_metadata.empty(),
                    "[gguf tokenizer] empty GGUF tokenizer metadata: there is nothing to build a tokenizer from.");
    auto tokenizer_config = tokenizer_config_from_rt_info(tokenizer_metadata);
    return build_tokenizer_models(shared_object_ov_tokenizers, std::move(tokenizer_config));
}

#ifdef ENABLE_GGUF
namespace {
std::shared_ptr<ov::frontend::gguf::GGUFTokenizerMetadata> find_tokenizer_metadata(
    const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "[gguf tokenizer] null model: cannot read GGUF tokenizer metadata from it.");
    const auto& rt = model->get_rt_info();
    auto it = rt.find(ov::frontend::gguf::gguf_tokenizer_metadata_key());
    OPENVINO_ASSERT(it != rt.end(),
                    "[gguf tokenizer] the model has no '",
                    ov::frontend::gguf::gguf_tokenizer_metadata_key(),
                    "' runtime info, so no tokenizer can be built from it. Only a model converted from a .gguf "
                    "by the OpenVINO GGUF frontend carries it; a plain IR, a model whose rt_info was stripped, "
                    "or one built by the legacy GGUF reader does not. Build the Tokenizer from the .gguf path "
                    "instead.");
    auto attr = it->second.as<std::shared_ptr<ov::frontend::gguf::GGUFTokenizerMetadata>>();
    OPENVINO_ASSERT(attr, "[gguf tokenizer] unexpected type for the GGUF tokenizer metadata runtime info.");
    return attr;
}
}  // namespace

const ov::AnyMap& gguf_tokenizer_metadata_from_model(const std::shared_ptr<ov::Model>& model) {
    return find_tokenizer_metadata(model)->config;
}

ov::AnyMap take_gguf_tokenizer_metadata(const std::shared_ptr<ov::Model>& model) {
    auto attr = find_tokenizer_metadata(model);
    ov::AnyMap config = std::move(attr->config);
    erase_gguf_tokenizer_metadata(model);
    return config;
}

void erase_gguf_tokenizer_metadata(const std::shared_ptr<ov::Model>& model) {
    if (model) {
        model->get_rt_info().erase(ov::frontend::gguf::gguf_tokenizer_metadata_key());
    }
}
#else
const ov::AnyMap& gguf_tokenizer_metadata_from_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
}

ov::AnyMap take_gguf_tokenizer_metadata(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
}

void erase_gguf_tokenizer_metadata(const std::shared_ptr<ov::Model>& model) {}
#endif

std::function<std::string(const std::string&)> gguf_chat_template_patcher(
    const std::map<std::string, GGUFMetaData>& tokenizer_config) {
    // Qwen3.5 templates test "enable_thinking is undefined"; reuse the Qwen3.5 VLM patch.
    if (const auto* pre = get_if_exist<std::string>(tokenizer_config, "pre"); pre && *pre == "qwen35") {
        return utils::patch_chat_template_undefined_tests;
    }
    return {};
}

std::string patch_gguf_chat_template(const std::string& chat_template) {
    std::string patched_chat_template = chat_template;
    // Define the exact pattern to find in original chat_template
    // Using C++ raw string literals (R"(...)") to correctly represent the literal content,
    const std::string qwen2_5_substring_to_find = R"({{\"name\": <function-name>, \"arguments\": <args-json-object>}})";
    // Define the exact replacement substring for str2
    const std::string qwen2_5_replacement_substring =
        R"({\"name\": <function-name>, \"arguments\": <args-json-object>})";
    // Find the position of the substring to be replaced
    size_t pos_qwen2_5 = patched_chat_template.find(qwen2_5_substring_to_find);
    if (pos_qwen2_5 != std::string::npos) {
        // Substring found, perform the replacement
        patched_chat_template.replace(pos_qwen2_5, qwen2_5_substring_to_find.length(), qwen2_5_replacement_substring);
    }

    const std::string qwen3_substring_to_find_0 = R"({%- for index in range(ns.last_query_index, -1, -1) %})";
    const std::string qwen3_substring_to_find_1 = R"({%- set message = messages[index] %})";
    const std::string qwen3_substring_to_find_2 = R"({%- if ns.multi_step_tool and message.role == "user" and not('<tool_response>' in message.content and '</tool_response>' in message.content) %})";

    const std::string qwen3_replacement_substring_0 = R"({%- for message in messages[::-1] %})";
    const std::string qwen3_replacement_substring_1 = R"({%- set index = (messages|length - 1) - loop.index0 %})";
    const std::string qwen3_replacement_substring_2 = R"({%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %})";

    const std::string qwen3_substring_to_find = qwen3_substring_to_find_0 + "\n" + "    " + qwen3_substring_to_find_1 + "\n" + "    "  + qwen3_substring_to_find_2;
    const std::string qwen3_replacement_substring = qwen3_replacement_substring_0 + "\n" + "    " + qwen3_replacement_substring_1 + "\n" + "    "  + qwen3_replacement_substring_2;
    size_t pos_qwen3 = patched_chat_template.find(qwen3_substring_to_find);
    
    if (pos_qwen3 != std::string::npos) {
        // Substring found, perform the replacement
        patched_chat_template.replace(pos_qwen3, qwen3_substring_to_find.length(), qwen3_replacement_substring);
    }

    return patched_chat_template;
}

}  // namespace genai
}  // namespace ov
