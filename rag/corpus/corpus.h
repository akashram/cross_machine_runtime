#pragma once

// Shared fixture for Phase 13 steps 4/6/7/8: a small set of factual
// documents plus a matching, hand-labeled query set (relevance judgments +
// the exact answer substring each query's document supports). One shared
// definition so every downstream step measures against the identical
// corpus rather than each inventing its own -- recall@k (step 6),
// generation-quality-with-vs-without-retrieval (step 7), and the
// approximate-vs-exact system-level study (step 8) are only comparable to
// each other if they share ground truth.
//
// Scope note: each document is short enough (one or two sentences) that
// indexing_pipeline's default chunking settings produce exactly ONE chunk
// per document here -- so "relevant chunk" and "relevant document" mean
// the same thing throughout steps 6-8. indexing_pipeline_test.cpp
// exercises the sentence-boundary+overlap chunking logic itself
// separately, on a deliberately longer synthetic paragraph that DOES
// split into multiple chunks -- decoupling "does chunking work" from
// "does retrieval over this corpus work" keeps the bookkeeping in every
// later step tractable.

#include <string>
#include <vector>

namespace rag {

struct Document {
  int id;
  std::string text;
};

struct QueryJudgment {
  std::string query;
  int relevant_doc_id;     // this corpus: exactly one relevant document per query
  std::string answer;      // lowercase substring the correct document supports
};

inline std::vector<Document> sample_documents() {
  return {
      {0, "paris is the capital of france. it is known for the eiffel tower."},
      {1, "jupiter is the largest planet in the solar system. it is a gas giant."},
      {2, "water boils at 100 degrees celsius at sea level. steam forms above it."},
      {3, "the great wall of china is thousands of miles long. it was built over centuries."},
      {4, "mount everest is the tallest mountain on earth. it is located in the himalayas."},
      {5, "plants use photosynthesis to make food from sunlight. oxygen is released as a byproduct."},
      {6, "the sun is a star at the center of the solar system. it provides heat and light to earth."},
      {7, "the pacific ocean is the largest ocean on earth. it is deeper than the atlantic ocean."},
  };
}

// Bulk filler for steps 6/8: at only 8 documents, ml/knn's BallTree
// (default leaf_size=10) is a single leaf, so its "approximate" mode is
// literally identical to its exact mode -- no real ANN tradeoff to
// measure. These distractor documents carry no relevance label (every
// sample_queries() judgment still points at a sample_documents() id) and
// exist purely to bulk the index up to a scale where approximate search
// actually behaves differently from exact search, the same way ml/knn's
// own tests use n=300-2000 rather than a handful of points. Generated
// from real-word templates (not gibberish) so the encoder's char-level
// tokenizer still sees plausible English text, deterministic (no RNG) so
// every step sees the identical distractor set.
inline std::vector<Document> sample_distractor_documents(int count = 40) {
  static const char *kSubjects[] = {"the engineer",   "the artist",   "the farmer",     "the pilot",
                                     "the musician",   "the athlete",  "the chef",       "the teacher",
                                     "the scientist",  "the sailor"};
  static const char *kVerbs[] = {"builds", "studies", "repairs", "designs", "practices"};
  static const char *kObjects[] = {"a wooden table",   "a small engine",  "an old bridge",
                                    "a garden fence",   "a paper map",     "a metal frame",
                                    "a quiet melody",   "a running trail"};
  static const char *kTails[] = {"every morning before work.", "with great care and patience.",
                                  "during the long afternoon.", "after a short break for lunch."};

  std::vector<Document> docs;
  docs.reserve(static_cast<std::size_t>(count));
  int next_id = 8; // sample_documents() uses ids 0-7
  for (int i = 0; i < count; ++i) {
    std::string text = std::string(kSubjects[i % 10]) + " " + kVerbs[i % 5] + " " + kObjects[i % 8] + " " +
                        kTails[i % 4];
    docs.push_back({next_id++, text});
  }
  return docs;
}

inline std::vector<Document> sample_documents_with_distractors(int distractor_count = 40) {
  std::vector<Document> docs = sample_documents();
  std::vector<Document> distractors = sample_distractor_documents(distractor_count);
  docs.insert(docs.end(), distractors.begin(), distractors.end());
  return docs;
}

inline std::vector<QueryJudgment> sample_queries() {
  return {
      {"what is the capital of france", 0, "paris"},
      {"what is the largest planet in the solar system", 1, "jupiter"},
      {"what temperature does water boil at sea level", 2, "100"},
      {"which country has the great wall", 3, "china"},
      {"what is the tallest mountain on earth", 4, "everest"},
      {"what gas do plants release during photosynthesis", 5, "oxygen"},
      {"what is at the center of the solar system", 6, "sun"},
      {"what is the largest ocean on earth", 7, "pacific"},
  };
}

} // namespace rag
