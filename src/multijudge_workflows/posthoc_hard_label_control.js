"use strict";

const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const SCRIPT_ROOT = __dirname;
const V15_ROOT = path.resolve(SCRIPT_ROOT, "..", "..");
const RESULT_ROOT = path.join(V15_ROOT, "results", "mmds");
const CONFIG_ROOT = path.join(SCRIPT_ROOT, "configs");
const JUDGES = ["qwen", "safework", "minicpm"];
const EPS = 1e-6;

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function readJsonl(file) {
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function sha256File(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, file);
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function logit(p) {
  const q = Math.min(1 - EPS, Math.max(EPS, p));
  return Math.log(q / (1 - q));
}

function mean(values) {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function variance(values) {
  const m = mean(values);
  return mean(values.map((v) => (v - m) ** 2));
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * q;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  if (low === high) return sorted[low];
  const fraction = position - low;
  return sorted[low] * (1 - fraction) + sorted[high] * fraction;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function groupCounts(groups) {
  const counts = new Map();
  for (const group of groups) counts.set(group, (counts.get(group) || 0) + 1);
  return counts;
}

function clusterWeights(groups) {
  const counts = groupCounts(groups);
  const groupCount = counts.size;
  return groups.map((group) => groups.length / (groupCount * counts.get(group)));
}

function solveLinear(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    if (Math.abs(a[pivot][col]) < 1e-14) throw new Error("Singular Newton system");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j += 1) a[col][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map((row) => row[n]);
}

function fitL2Logistic(features, targets, groups, cValue) {
  const n = targets.length;
  const d = features[0].length;
  const weights = clusterWeights(groups);
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const featureMean = Array(d).fill(0);
  for (let j = 0; j < d; j += 1) {
    for (let i = 0; i < n; i += 1) featureMean[j] += weights[i] * features[i][j];
    featureMean[j] /= weightSum;
  }
  const scale = Array(d).fill(0);
  for (let j = 0; j < d; j += 1) {
    for (let i = 0; i < n; i += 1) scale[j] += weights[i] * (features[i][j] - featureMean[j]) ** 2;
    scale[j] = Math.sqrt(Math.max(scale[j] / weightSum, 1e-12));
  }
  const x = features.map((row) => row.map((value, j) => (value - featureMean[j]) / scale[j]));
  let weightedPositive = 0;
  for (let i = 0; i < n; i += 1) weightedPositive += weights[i] * targets[i];
  const prevalence = Math.min(1 - EPS, Math.max(EPS, weightedPositive / weightSum));
  let beta = [logit(prevalence), ...Array(d).fill(0)];

  function objective(candidate) {
    let value = 0;
    for (let i = 0; i < n; i += 1) {
      let eta = candidate[0];
      for (let j = 0; j < d; j += 1) eta += candidate[j + 1] * x[i][j];
      const p = Math.min(1 - EPS, Math.max(EPS, sigmoid(eta)));
      value -= weights[i] * (targets[i] * Math.log(p) + (1 - targets[i]) * Math.log(1 - p));
    }
    for (let j = 1; j <= d; j += 1) value += 0.5 * betaPenalty(candidate[j], cValue);
    return value;
  }

  let iterations = 0;
  for (; iterations < 200; iterations += 1) {
    const gradient = Array(d + 1).fill(0);
    const hessian = Array.from({ length: d + 1 }, () => Array(d + 1).fill(0));
    for (let i = 0; i < n; i += 1) {
      const row = [1, ...x[i]];
      let eta = 0;
      for (let j = 0; j <= d; j += 1) eta += beta[j] * row[j];
      const p = sigmoid(eta);
      const residual = weights[i] * (p - targets[i]);
      const curvature = weights[i] * Math.max(p * (1 - p), 1e-12);
      for (let j = 0; j <= d; j += 1) {
        gradient[j] += residual * row[j];
        for (let k = 0; k <= d; k += 1) hessian[j][k] += curvature * row[j] * row[k];
      }
    }
    for (let j = 1; j <= d; j += 1) {
      gradient[j] += beta[j] / cValue;
      hessian[j][j] += 1 / cValue;
    }
    const step = solveLinear(hessian, gradient);
    const before = objective(beta);
    let factor = 1;
    let candidate;
    while (factor > 1e-8) {
      candidate = beta.map((value, j) => value - factor * step[j]);
      if (objective(candidate) <= before + 1e-12) break;
      factor /= 2;
    }
    if (factor <= 1e-8) break;
    beta = candidate;
    if (Math.max(...step.map((value) => Math.abs(factor * value))) < 1e-10) break;
  }
  return {
    mean: featureMean,
    scale,
    coefficients: beta.slice(1),
    intercept: beta[0],
    penalty: "l2",
    C: cValue,
    iterations,
  };
}

function betaPenalty(value, cValue) {
  return (value * value) / cValue;
}

function predictModel(model, features) {
  return features.map((row) => {
    let eta = model.intercept;
    for (let j = 0; j < row.length; j += 1) {
      eta += model.coefficients[j] * ((row[j] - model.mean[j]) / model.scale[j]);
    }
    return sigmoid(eta);
  });
}

function makeGroupFolds(targets, groups, nSplits, seed) {
  const unique = [...new Set(groups)].sort();
  const random = mulberry32(seed);
  const tie = new Map(unique.map((group) => [group, random()]));
  const indices = new Map(unique.map((group) => [group, []]));
  groups.forEach((group, i) => indices.get(group).push(i));
  const summaries = unique.map((group) => {
    const ids = indices.get(group);
    const positive = ids.reduce((sum, i) => sum + targets[i], 0);
    return { group, positive, negative: ids.length - positive, size: ids.length, tie: tie.get(group) };
  });
  summaries.sort((a, b) => {
    const keysA = [a.positive > 0 ? 1 : 0, a.positive, a.size];
    const keysB = [b.positive > 0 ? 1 : 0, b.positive, b.size];
    for (let i = 0; i < keysA.length; i += 1) if (keysA[i] !== keysB[i]) return keysB[i] - keysA[i];
    if (a.tie !== b.tie) return a.tie - b.tie;
    return a.group.localeCompare(b.group);
  });
  const foldPositive = Array(nSplits).fill(0);
  const foldNegative = Array(nSplits).fill(0);
  const foldRecords = Array(nSplits).fill(0);
  const foldGroups = Array(nSplits).fill(0);
  const targetPositive = Math.max(targets.reduce((a, b) => a + b, 0) / nSplits, 1);
  const targetNegative = Math.max((targets.length - targets.reduce((a, b) => a + b, 0)) / nSplits, 1);
  const targetRecords = Math.max(targets.length / nSplits, 1);
  const targetGroups = Math.max(unique.length / nSplits, 1);
  const assignment = new Map();
  for (const summary of summaries) {
    const candidates = [];
    for (let fold = 0; fold < nSplits; fold += 1) {
      const p = [...foldPositive]; p[fold] += summary.positive;
      const q = [...foldNegative]; q[fold] += summary.negative;
      const r = [...foldRecords]; r[fold] += summary.size;
      const g = [...foldGroups]; g[fold] += 1;
      const score = variance(p.map((v) => v / targetPositive)) + variance(q.map((v) => v / targetNegative)) +
        0.25 * variance(r.map((v) => v / targetRecords)) + 0.05 * variance(g.map((v) => v / targetGroups));
      candidates.push({ score, records: foldRecords[fold], groups: foldGroups[fold], fold });
    }
    candidates.sort((a, b) => a.score - b.score || a.records - b.records || a.groups - b.groups || a.fold - b.fold);
    const selected = candidates[0].fold;
    assignment.set(summary.group, selected);
    foldPositive[selected] += summary.positive;
    foldNegative[selected] += summary.negative;
    foldRecords[selected] += summary.size;
    foldGroups[selected] += 1;
  }
  return groups.map((group) => assignment.get(group));
}

function classificationMetrics(targets, predictions) {
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  for (let i = 0; i < targets.length; i += 1) {
    if (targets[i] === 1 && predictions[i] === 1) tp += 1;
    else if (targets[i] === 0 && predictions[i] === 0) tn += 1;
    else if (targets[i] === 0 && predictions[i] === 1) fp += 1;
    else fn += 1;
  }
  const unsafeF1 = 2 * tp + fp + fn === 0 ? 0 : (2 * tp) / (2 * tp + fp + fn);
  const safeF1 = 2 * tn + fn + fp === 0 ? 0 : (2 * tn) / (2 * tn + fn + fp);
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  const specificity = tn + fp === 0 ? 0 : tn / (tn + fp);
  const denom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return {
    sample_count: targets.length,
    accuracy: (tp + tn) / targets.length,
    macro_f1: (unsafeF1 + safeF1) / 2,
    unsafe_recall: recall,
    safe_specificity: specificity,
    false_safe_rate: 1 - recall,
    mcc: denom === 0 ? 0 : (tp * tn - fp * fn) / denom,
    unsafe_true_positive: tp,
    safe_true_positive: tn,
    unsafe_false_positive: fp,
    unsafe_false_negative: fn,
  };
}

function selectConstrainedThreshold(targets, probabilities, minRecall, minSpecificity) {
  const unique = [...new Set(probabilities.map((p) => Math.min(1, Math.max(0, p))))].sort((a, b) => a - b);
  const candidates = [0, 0.5, 1, 1 + Number.EPSILON, ...unique];
  for (let i = 0; i + 1 < unique.length; i += 1) candidates.push((unique[i] + unique[i + 1]) / 2);
  const deduped = [...new Set(candidates)].sort((a, b) => a - b);
  const feasible = [];
  for (const threshold of deduped) {
    const metrics = classificationMetrics(targets, probabilities.map((p) => p >= threshold ? 1 : 0));
    if (metrics.unsafe_recall + 1e-12 >= minRecall && metrics.safe_specificity + 1e-12 >= minSpecificity) {
      feasible.push({ threshold, metrics });
    }
  }
  feasible.sort((a, b) => b.metrics.macro_f1 - a.metrics.macro_f1 || b.metrics.unsafe_recall - a.metrics.unsafe_recall ||
    b.metrics.safe_specificity - a.metrics.safe_specificity || Math.abs(a.threshold - 0.5) - Math.abs(b.threshold - 0.5) || a.threshold - b.threshold);
  if (!feasible.length) return { feasible: false, minimum_unsafe_recall: minRecall, minimum_safe_specificity: minSpecificity };
  return { feasible: true, ...feasible[0], minimum_unsafe_recall: minRecall, minimum_safe_specificity: minSpecificity, candidate_count: deduped.length };
}

function probabilityMetrics(targets, probabilities, eceBins = 10) {
  let brier = 0; let logLoss = 0; let ece = 0; let populated = 0;
  for (let i = 0; i < targets.length; i += 1) {
    const p = probabilities[i];
    brier += (p - targets[i]) ** 2;
    const clipped = Math.min(1 - EPS, Math.max(EPS, p));
    logLoss -= targets[i] * Math.log(clipped) + (1 - targets[i]) * Math.log(1 - clipped);
  }
  for (let bin = 0; bin < eceBins; bin += 1) {
    const low = bin / eceBins; const high = (bin + 1) / eceBins;
    const ids = probabilities.map((p, i) => ({ p, i })).filter(({ p }) => p >= low && (bin === eceBins - 1 ? p <= high : p < high));
    if (!ids.length) continue;
    populated += 1;
    const conf = mean(ids.map(({ p }) => p));
    const freq = mean(ids.map(({ i }) => targets[i]));
    ece += (ids.length / targets.length) * Math.abs(conf - freq);
  }
  return { sample_count: targets.length, brier_score: brier / targets.length, log_loss: logLoss / targets.length, ece, ece_bins: eceBins, populated_ece_bins: populated };
}

function groupedIndices(groups) {
  const map = new Map();
  groups.forEach((group, i) => { if (!map.has(group)) map.set(group, []); map.get(group).push(i); });
  return map;
}

function pairedBootstrap(targets, groups, leftPred, rightPred, leftProb, rightProb, replicates, seed) {
  const grouped = groupedIndices(groups);
  const components = [...grouped.keys()].sort();
  const random = mulberry32(seed);
  const distributions = { macro_f1: [], unsafe_recall: [], safe_specificity: [], mcc: [], brier_score: [], log_loss: [] };
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    const ids = [];
    for (let k = 0; k < components.length; k += 1) {
      const component = components[Math.floor(random() * components.length)];
      ids.push(...grouped.get(component));
    }
    const y = ids.map((i) => targets[i]);
    const lm = classificationMetrics(y, ids.map((i) => leftPred[i]));
    const rm = classificationMetrics(y, ids.map((i) => rightPred[i]));
    for (const name of ["macro_f1", "unsafe_recall", "safe_specificity", "mcc"]) distributions[name].push(lm[name] - rm[name]);
    const lp = probabilityMetrics(y, ids.map((i) => leftProb[i]));
    const rp = probabilityMetrics(y, ids.map((i) => rightProb[i]));
    distributions.brier_score.push(lp.brier_score - rp.brier_score);
    distributions.log_loss.push(lp.log_loss - rp.log_loss);
  }
  const metrics = {};
  for (const [name, values] of Object.entries(distributions)) {
    metrics[name] = { mean_delta: mean(values), median_delta: quantile(values, 0.5), ci95: [quantile(values, 0.025), quantile(values, 0.975)], positive_fraction: mean(values.map((v) => v > 0 ? 1 : 0)) };
  }
  return { schema: "paired-component-bootstrap-probability-v1", replicates, seed, component_count: components.length, difference_direction: "left_minus_right", metrics };
}

function pairedRandomization(targets, groups, left, right, replicates, seed) {
  const grouped = groupedIndices(groups);
  const components = [...grouped.keys()].sort();
  const random = mulberry32(seed);
  const observed = classificationMetrics(targets, left).macro_f1 - classificationMetrics(targets, right).macro_f1;
  let greater = 0; let greaterAbs = 0;
  for (let replicate = 0; replicate < replicates; replicate += 1) {
    const l = [...left]; const r = [...right];
    for (const component of components) {
      if (random() < 0.5) for (const i of grouped.get(component)) [l[i], r[i]] = [r[i], l[i]];
    }
    const delta = classificationMetrics(targets, l).macro_f1 - classificationMetrics(targets, r).macro_f1;
    if (delta >= observed) greater += 1;
    if (Math.abs(delta) >= Math.abs(observed)) greaterAbs += 1;
  }
  return { schema: "paired-component-randomization-v1", replicates, seed, component_count: components.length, metric: "macro_f1", observed_delta: observed, one_sided_p_value: (1 + greater) / (replicates + 1), two_sided_p_value: (1 + greaterAbs) / (replicates + 1) };
}

function applyCalibrator(parameters, value) {
  const p = Math.min(1 - EPS, Math.max(EPS, value));
  if (parameters.method === "identity") return p;
  if (parameters.method === "temperature") return sigmoid(logit(p) / parameters.temperature);
  if (parameters.method === "beta") return sigmoid(parameters.a * Math.log(p) - parameters.b * Math.log(1 - p) + parameters.c);
  throw new Error(`Unknown calibrator ${parameters.method}`);
}

function loadDevelopmentSplit(split) {
  const manifestPath = path.join(CONFIG_ROOT, `softfusion_${split}.jsonl`);
  const manifest = readJsonl(manifestPath);
  const byJudge = {};
  for (const judge of JUDGES) {
    const file = path.join(RESULT_ROOT, "qualification", judge, split, "judgments.jsonl");
    const records = new Map(readJsonl(file).map((record) => [record.sample_id, record]));
    byJudge[judge] = manifest.map((record) => records.get(record.sample_id).soft_evidence.p_unsafe);
  }
  return {
    manifestPath,
    sampleIds: manifest.map((record) => record.sample_id),
    targets: manifest.map((record) => record.assistant_label === "unsafe" ? 1 : 0),
    groups: manifest.map((record) => record.component_id),
    raw: byJudge,
  };
}

function run() {
  const fit = loadDevelopmentSplit("fit");
  const select = loadDevelopmentSplit("select");
  const targets = [...fit.targets, ...select.targets];
  const groups = [...fit.groups.map((g) => `fit:${g}`), ...select.groups.map((g) => `select:${g}`)];
  const hardFeatures = targets.map((_, i) => JUDGES.map((judge) => {
    const value = i < fit.targets.length ? fit.raw[judge][i] : select.raw[judge][i - fit.targets.length];
    return value >= 0.5 ? 1 : 0;
  }));

  const config = readJson(path.join(RESULT_ROOT, "fit", "fusion_config.json"));
  const cValue = config.model.C;
  const folds = makeGroupFolds(targets, groups, 5, config.seed + 3000000);
  const oof = Array(targets.length).fill(NaN);
  const foldAudit = [];
  for (let fold = 0; fold < 5; fold += 1) {
    const training = targets.map((_, i) => i).filter((i) => folds[i] !== fold);
    const validation = targets.map((_, i) => i).filter((i) => folds[i] === fold);
    const model = fitL2Logistic(training.map((i) => hardFeatures[i]), training.map((i) => targets[i]), training.map((i) => groups[i]), cValue);
    const predictions = predictModel(model, validation.map((i) => hardFeatures[i]));
    validation.forEach((i, k) => { oof[i] = predictions[k]; });
    foldAudit.push({ fold, training_record_count: training.length, validation_record_count: validation.length, training_component_count: new Set(training.map((i) => groups[i])).size, validation_component_count: new Set(validation.map((i) => groups[i])).size });
  }
  if (oof.some((value) => !Number.isFinite(value))) throw new Error("OOF predictions incomplete");
  const baselineSelection = readJson(path.join(RESULT_ROOT, "fit", "baseline_selection.json"));
  const reference = baselineSelection.development_metrics[baselineSelection.strongest_simple_baseline];
  const threshold = selectConstrainedThreshold(targets, oof, Math.max(0, reference.unsafe_recall - 0.01), Math.max(0, reference.safe_specificity - 0.03));
  if (!threshold.feasible) throw new Error("Hard-label control threshold infeasible");
  const finalModel = fitL2Logistic(hardFeatures, targets, groups, cValue);

  const confirmPath = path.join(RESULT_ROOT, "evaluation", "confirmatory_predictions.jsonl");
  const confirm = readJsonl(confirmPath);
  const confirmTargets = confirm.map((record) => record.expected_label === "unsafe" ? 1 : 0);
  const confirmGroups = confirm.map((record) => record.component_id);
  const confirmHard = confirm.map((record) => JUDGES.map((judge) => record.raw_p_unsafe[judge] >= 0.5 ? 1 : 0));
  const hardProbabilities = predictModel(finalModel, confirmHard);
  const hardPredictions = hardProbabilities.map((p) => p >= threshold.threshold ? 1 : 0);
  const fusionProbabilities = confirm.map((record) => record.fusion_p_unsafe);
  const fusionPredictions = confirm.map((record) => record.fusion_prediction === "unsafe" ? 1 : 0);
  const softMeanProbabilities = confirm.map((record) => mean(JUDGES.map((judge) => record.calibrated_p_unsafe[judge])));
  const analysisPlan = readJson(path.join(RESULT_ROOT, "fit", "confirmatory_analysis_plan.json"));
  const softMeanPredictions = softMeanProbabilities.map((p) => p >= analysisPlan.calibrated_soft_mean.threshold ? 1 : 0);

  const hardComparison = {
    point_differences: (() => {
      const left = classificationMetrics(confirmTargets, fusionPredictions);
      const right = classificationMetrics(confirmTargets, hardPredictions);
      return Object.fromEntries(["macro_f1", "unsafe_recall", "safe_specificity", "mcc"].map((name) => [name, left[name] - right[name]]));
    })(),
    bootstrap: pairedBootstrap(confirmTargets, confirmGroups, fusionPredictions, hardPredictions, fusionProbabilities, hardProbabilities, 10000, 202608071),
    randomization: pairedRandomization(confirmTargets, confirmGroups, fusionPredictions, hardPredictions, 10000, 202608072),
  };
  const softMeanProbabilityComparison = pairedBootstrap(confirmTargets, confirmGroups, fusionPredictions, softMeanPredictions, fusionProbabilities, softMeanProbabilities, 10000, 202608073);

  const calibratedFeatures = targets.map((_, i) => JUDGES.map((judge) => {
    const raw = i < fit.targets.length ? fit.raw[judge][i] : select.raw[judge][i - fit.targets.length];
    return logit(applyCalibrator(config.final_development_calibrators[judge], raw));
  }));
  const verificationModel = fitL2Logistic(calibratedFeatures, targets, groups, cValue);
  const maxDiff = Math.max(
    Math.abs(verificationModel.intercept - config.model.intercept),
    ...verificationModel.coefficients.map((value, i) => Math.abs(value - config.model.coefficients[i])),
    ...verificationModel.mean.map((value, i) => Math.abs(value - config.model.mean[i])),
    ...verificationModel.scale.map((value, i) => Math.abs(value - config.model.scale[i])),
  );

  const stored = readJson(path.join(RESULT_ROOT, "evaluation", "confirmatory_results.json"));
  const holmFamily = Object.entries(stored.comparisons).filter(([name]) => name !== "majority").map(([name, value]) => ({ comparator: name, raw_one_sided_p_value: value.randomization.one_sided_p_value, holm_adjusted_one_sided_p_value: value.holm_adjusted_one_sided_p_value }));

  const sourceFiles = [fit.manifestPath, select.manifestPath, confirmPath, path.join(RESULT_ROOT, "fit", "fusion_config.json"), path.join(RESULT_ROOT, "fit", "confirmatory_analysis_plan.json"), path.join(RESULT_ROOT, "evaluation", "confirmatory_results.json")];
  for (const judge of JUDGES) for (const split of ["fit", "select"]) sourceFiles.push(path.join(RESULT_ROOT, "qualification", judge, split, "judgments.jsonl"));
  const result = {
    schema: "softfusion-posthoc-hard-label-control-v1",
    status: "completed",
    inferential_role: "post_hoc_mechanism_control_not_part_of_the_frozen_confirmatory_hierarchy",
    leakage_guard: {
      model_fit_data: "D_fit plus D_select development records only",
      threshold_selection_data: "five-fold component-grouped development OOF predictions only",
      confirmatory_labels_used_for_fit_or_threshold: false,
      confirmatory_labels_used_for_final_evaluation: true,
    },
    control_definition: {
      input_features: "three binary judge decisions obtained by thresholding each raw p_unsafe at 0.5",
      learner: "three-main-effect component-weighted L2 logistic regression",
      C: cValue,
      feature_count: 3,
      fitted_parameter_count: 4,
      comparator_difference: "input information only; learner family, regularization strength, development split, component grouping, risk constraints, and threshold-selection rule match the frozen F1 pipeline",
    },
    implementation_verification: {
      check: "Node Newton solver refit on the frozen continuous features versus stored scikit-learn model",
      maximum_absolute_parameter_or_scaling_difference: maxDiff,
      passed: maxDiff < 1e-6,
    },
    development: {
      record_count: targets.length,
      component_count: new Set(groups).size,
      fold_audit: foldAudit,
      threshold,
      final_model: finalModel,
    },
    confirmatory: {
      record_count: confirmTargets.length,
      component_count: new Set(confirmGroups).size,
      continuous_fusion_metrics: classificationMetrics(confirmTargets, fusionPredictions),
      hard_label_logistic_metrics: classificationMetrics(confirmTargets, hardPredictions),
      continuous_fusion_probability_metrics: probabilityMetrics(confirmTargets, fusionProbabilities),
      hard_label_logistic_probability_metrics: probabilityMetrics(confirmTargets, hardProbabilities),
      fusion_minus_hard_label_control: hardComparison,
      fusion_minus_calibrated_soft_mean_probability_bootstrap: softMeanProbabilityComparison,
    },
    multiple_testing_clarification: {
      primary_majority_comparison_excluded_from_holm_family: true,
      secondary_family_size: holmFamily.length,
      family: holmFamily,
    },
    ece_definition: "10 fixed-width bins on [0,1], left-closed and right-open except the final right-closed bin; record-weighted absolute calibration gaps",
    source_files_sha256: Object.fromEntries(sourceFiles.map((file) => [path.relative(V15_ROOT, file).replaceAll("\\", "/"), sha256File(file)])),
  };
  const output = path.join(RESULT_ROOT, "evaluation", "posthoc_hard_label_control.json");
  writeJson(output, result);
  console.log(JSON.stringify({ output, verification_passed: result.implementation_verification.passed, hard_label_macro_f1: result.confirmatory.hard_label_logistic_metrics.macro_f1, continuous_macro_f1: result.confirmatory.continuous_fusion_metrics.macro_f1, delta: result.confirmatory.fusion_minus_hard_label_control.point_differences.macro_f1, ci95: result.confirmatory.fusion_minus_hard_label_control.bootstrap.metrics.macro_f1.ci95, probability_ci_brier: result.confirmatory.fusion_minus_calibrated_soft_mean_probability_bootstrap.metrics.brier_score.ci95, probability_ci_log_loss: result.confirmatory.fusion_minus_calibrated_soft_mean_probability_bootstrap.metrics.log_loss.ci95 }, null, 2));
}

run();
