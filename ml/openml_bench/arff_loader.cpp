#include "arff_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

std::string trim(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string strip_quotes(const std::string& s) {
    std::string t = trim(s);
    if (t.size() >= 2 && ((t.front() == '\'' && t.back() == '\'') || (t.front() == '"' && t.back() == '"')))
        return t.substr(1, t.size() - 2);
    return t;
}

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool starts_with_ci(const std::string& s, const std::string& prefix) {
    if (s.size() < prefix.size()) return false;
    return to_lower(s.substr(0, prefix.size())) == prefix;
}

struct Attribute {
    bool is_nominal = false;
    std::vector<std::string> categories;  // only used if is_nominal
};

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    for (char c : line) {
        if (c == ',') {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    fields.push_back(trim(current));
    return fields;
}

}  // namespace

OpenMLDataset load_arff(const std::string& path, const std::string& display_name) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("could not open ARFF file: " + path);

    std::vector<Attribute> attributes;
    bool in_data_section = false;

    Features raw_rows;
    std::vector<std::string> class_tokens;

    std::string line;
    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '%') continue;

        if (!in_data_section && starts_with_ci(trimmed, "@attribute")) {
            std::string rest = trim(trimmed.substr(std::string("@attribute").size()));
            // Skip the (possibly quoted, possibly space-containing) name
            // by finding the type specification, which either starts
            // with '{' (nominal) or is the last whitespace-delimited
            // token (numeric type keyword: real/numeric/integer).
            std::size_t brace_pos = rest.find('{');
            Attribute attr;
            if (brace_pos != std::string::npos) {
                attr.is_nominal = true;
                std::size_t close = rest.find('}', brace_pos);
                std::string inner = rest.substr(brace_pos + 1, close - brace_pos - 1);
                std::stringstream ss(inner);
                std::string token;
                while (std::getline(ss, token, ',')) attr.categories.push_back(strip_quotes(token));
            }
            attributes.push_back(attr);
        } else if (!in_data_section && starts_with_ci(trimmed, "@data")) {
            in_data_section = true;
        } else if (in_data_section) {
            std::vector<std::string> fields = split_csv_line(trimmed);
            if (fields.size() != attributes.size()) continue;  // malformed/truncated row: skip

            std::vector<float> row(fields.size() - 1);
            for (std::size_t f = 0; f + 1 < fields.size(); ++f) {
                if (fields[f] == "?")
                    row[f] = std::numeric_limits<float>::quiet_NaN();
                else
                    row[f] = std::stof(fields[f]);
            }
            raw_rows.push_back(std::move(row));
            class_tokens.push_back(strip_quotes(fields.back()));
        }
    }

    if (attributes.empty() || !attributes.back().is_nominal)
        throw std::runtime_error("ARFF file's last attribute must be nominal (the class column): " + path);

    OpenMLDataset dataset;
    dataset.name = display_name;
    dataset.class_names = attributes.back().categories;

    std::size_t n = raw_rows.size();
    std::size_t d = n ? raw_rows[0].size() : 0;

    // Mean-impute missing ('?') values -- a documented, simple choice
    // (see README's Design note), not the most sophisticated missing-
    // data handling available.
    std::vector<float> col_sum(d, 0.0f);
    std::vector<int> col_count(d, 0);
    for (const auto& row : raw_rows)
        for (std::size_t f = 0; f < d; ++f)
            if (!std::isnan(row[f])) {
                col_sum[f] += row[f];
                ++col_count[f];
            }
    std::vector<float> col_mean(d, 0.0f);
    for (std::size_t f = 0; f < d; ++f) col_mean[f] = col_count[f] > 0 ? col_sum[f] / static_cast<float>(col_count[f]) : 0.0f;

    dataset.X.resize(n);
    dataset.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        dataset.X[i] = raw_rows[i];
        for (std::size_t f = 0; f < d; ++f)
            if (std::isnan(dataset.X[i][f])) dataset.X[i][f] = col_mean[f];

        auto it = std::find(dataset.class_names.begin(), dataset.class_names.end(), class_tokens[i]);
        if (it == dataset.class_names.end()) throw std::runtime_error("unrecognized class value '" + class_tokens[i] + "' in " + path);
        dataset.y[i] = static_cast<float>(it - dataset.class_names.begin());
    }

    return dataset;
}

TrainTestSplit split_train_test(const Features& X, const Labels& y, float test_fraction, unsigned random_state) {
    std::vector<std::size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_state);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::size_t n_test = static_cast<std::size_t>(static_cast<float>(X.size()) * test_fraction);
    TrainTestSplit split;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (i < n_test) {
            split.X_test.push_back(X[indices[i]]);
            split.y_test.push_back(y[indices[i]]);
        } else {
            split.X_train.push_back(X[indices[i]]);
            split.y_train.push_back(y[indices[i]]);
        }
    }
    return split;
}
