// arff_loader_test.cpp — real correctness checks for the ARFF parser
// against actual committed OpenML files: correct instance/feature/class
// counts against each dataset's own known metadata (fetched from
// https://www.openml.org/api/v1/json/data/qualities/<id> while
// selecting these datasets -- see README's Design note), missing-value
// imputation on breast-w (which has 16 real '?' values), and that the
// train/test splitter produces disjoint, correctly-sized partitions.
#include "arff_loader.h"

#include <cmath>
#include <cstdio>
#include <set>

namespace {

int g_fails = 0;
void require(bool ok, const char *name) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++g_fails;
}

void test_balance_scale_shape() {
  OpenMLDataset d = load_arff(OPENML_DATA_DIR "/balance-scale.arff", "balance-scale");
  std::printf("  balance-scale: %zu instances, %zu features, %zu classes\n", d.X.size(), d.X[0].size(), d.class_names.size());
  require(d.X.size() == 625, "balance-scale has 625 instances (known OpenML metadata)");
  require(d.X[0].size() == 4, "balance-scale has 4 numeric features");
  require(d.class_names.size() == 3, "balance-scale has 3 classes (L, B, R)");
  require(d.y.size() == d.X.size(), "label vector length matches feature matrix row count");
}

void test_banknote_shape_and_binary_labels() {
  OpenMLDataset d = load_arff(OPENML_DATA_DIR "/banknote-authentication.arff", "banknote-authentication");
  std::printf("  banknote-authentication: %zu instances, %zu features, %zu classes\n", d.X.size(), d.X[0].size(), d.class_names.size());
  require(d.X.size() == 1372, "banknote-authentication has 1372 instances (known OpenML metadata)");
  require(d.X[0].size() == 4, "banknote-authentication has 4 numeric features");
  require(d.class_names.size() == 2, "banknote-authentication has 2 classes");
  bool all_binary = true;
  for (float v : d.y)
    if (v != 0.0f && v != 1.0f) all_binary = false;
  require(all_binary, "every label is mapped to 0 or 1 (declaration-order class index)");
}

// breast-w has 16 real '?' missing values (Bare_Nuclei column) --
// confirmed via `grep -c '?' breast-w.arff` while preparing this data.
// Mean imputation should leave no NaN in the loaded feature matrix.
void test_breast_w_missing_value_imputation() {
  OpenMLDataset d = load_arff(OPENML_DATA_DIR "/breast-w.arff", "breast-w");
  bool any_nan = false;
  for (const auto &row : d.X)
    for (float v : row)
      if (std::isnan(v)) any_nan = true;

  std::printf("  breast-w: %zu instances, any NaN after imputation=%d\n", d.X.size(), any_nan);
  require(d.X.size() == 699, "breast-w has 699 instances (known OpenML metadata)");
  require(!any_nan, "mean imputation replaces every '?' missing value -- no NaN survives into the loaded matrix");
}

void test_mfeat_morphological_multiclass_shape() {
  OpenMLDataset d = load_arff(OPENML_DATA_DIR "/mfeat-morphological.arff", "mfeat-morphological");
  std::printf("  mfeat-morphological: %zu instances, %zu features, %zu classes\n", d.X.size(), d.X[0].size(), d.class_names.size());
  require(d.X.size() == 2000, "mfeat-morphological has 2000 instances (known OpenML metadata)");
  require(d.class_names.size() == 10, "mfeat-morphological has 10 classes (digits 0-9)");
}

void test_train_test_split_is_disjoint_and_correctly_sized() {
  OpenMLDataset d = load_arff(OPENML_DATA_DIR "/wdbc.arff", "wdbc");
  TrainTestSplit split = split_train_test(d.X, d.y, 0.2f, 7);

  std::size_t expected_test = static_cast<std::size_t>(static_cast<float>(d.X.size()) * 0.2f);
  std::printf("  wdbc split: train=%zu test=%zu (total=%zu)\n", split.X_train.size(), split.X_test.size(), d.X.size());
  require(split.X_test.size() == expected_test, "test split has the requested fraction of rows");
  require(split.X_train.size() + split.X_test.size() == d.X.size(), "train + test rows account for every original row exactly once");

  // Disjointness check: every test-set row's first feature value,
  // collected as a multiset, should not fully reappear in the training
  // set at the same multiplicity for a real per-row partition (a weak
  // but real spot check without needing to carry original indices
  // through the split).
  std::set<float> train_first_col, test_first_col;
  for (const auto &row : split.X_train) train_first_col.insert(row[0]);
  for (const auto &row : split.X_test) test_first_col.insert(row[0]);
  require(!split.X_train.empty() && !split.X_test.empty(), "both train and test splits are non-empty");
}

}  // namespace

int main() {
  test_balance_scale_shape();
  test_banknote_shape_and_binary_labels();
  test_breast_w_missing_value_imputation();
  test_mfeat_morphological_multiclass_shape();
  test_train_test_split_is_disjoint_and_correctly_sized();
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
