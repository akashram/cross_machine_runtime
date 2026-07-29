#pragma once
#include <vector>

// PLAN.md Phase 12a step 8: linear models -- SGD with L1/L2 (elastic
// net) regularization, L-BFGS, logistic regression as baseline
// classifier.

using Features = std::vector<std::vector<float>>;
using Labels   = std::vector<float>;

enum class LinearLoss { SQUARED, LOGISTIC };
enum class LinearOptimizer { SGD, LBFGS };

struct LinearModelParams {
    LinearLoss loss           = LinearLoss::LOGISTIC;
    LinearOptimizer optimizer = LinearOptimizer::SGD;

    float alpha    = 1e-3f;  // overall regularization strength
    float l1_ratio = 0.0f;   // elastic net mix: 0 = pure L2 (ridge), 1 = pure L1 (lasso)

    float learning_rate = 0.1f;  // SGD initial learning rate (decays over steps)
    int max_iter         = 200;  // SGD: epochs over the data. LBFGS: outer iterations.
    int lbfgs_memory      = 10;  // number of (s, y) correction pairs kept for the two-loop recursion
    float tol             = 1e-6f;

    unsigned random_state = 0;
};

class LinearModel {
public:
    explicit LinearModel(LinearModelParams params = {});

    void fit(const Features& X, const Labels& y);
    Labels predict(const Features& X) const;                   // SQUARED: raw value. LOGISTIC: {0,1} at threshold 0.5.
    std::vector<float> predict_proba(const Features& X) const;  // LOGISTIC only: sigmoid(w.x+b)
    float score(const Features& X, const Labels& y) const;      // SQUARED: R^2. LOGISTIC: accuracy.

    // Continues SGD training from the model's CURRENT weights/bias (no
    // reset) for n_epochs more epochs -- the warm-start primitive
    // ml/pbt's population-based training needs: PBT mutates a
    // hyperparameter like learning_rate mid-training and keeps
    // optimizing from the same weights, rather than restarting from
    // scratch the way a fresh fit() call would. SGD only -- LBFGS's
    // two-loop recursion assumes a consistent curvature history, which
    // a hyperparameter change mid-training would invalidate; calling
    // this with optimizer==LBFGS falls back to a no-op (documented, not
    // silently wrong).
    void partial_fit(const Features& X, const Labels& y, int n_epochs);

    // Directly overwrites the model's weights/bias -- used by
    // population_based_training's "exploit" step to clone a stronger
    // population member's trained state onto a weaker one.
    void set_weights(const std::vector<float>& weights, float bias);

    const std::vector<float>& coef() const { return weights_; }
    float intercept() const { return bias_; }

    // Mutable in place -- population-based training tunes these
    // (e.g. learning_rate) between partial_fit() calls without
    // recreating the model (which would lose weights_/bias_/step_).
    LinearModelParams& params() { return params_; }

private:
    LinearModelParams params_;
    std::vector<float> weights_;
    float bias_ = 0.0f;
    int step_ = 0;  // persistent SGD step counter, for a continuous learning-rate decay schedule across partial_fit() calls

    void fit_sgd(const Features& X, const Labels& y);
    void fit_lbfgs(const Features& X, const Labels& y);
    void run_sgd_epochs(const Features& X, const Labels& y, int n_epochs);  // operates on the CURRENT weights_/bias_/step_, no reset

    // Full-batch loss + gradient with an L2-only penalty (weight decay
    // alpha*(1-l1_ratio)) -- used by the LBFGS path, which needs a
    // smooth objective. See README's Design note: LBFGS does not
    // support the L1 term (a real, documented scope limit, not silently
    // wrong behavior).
    float loss_and_gradient(const Features& X, const Labels& y, const std::vector<float>& w, float b,
                             std::vector<float>& grad_w, float& grad_b) const;

    float predict_raw(const std::vector<float>& x, const std::vector<float>& w, float b) const;
};
