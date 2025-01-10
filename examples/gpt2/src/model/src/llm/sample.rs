use candle::{DType, Result, Tensor};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn sample_argmax(logits: Tensor) -> Result<u32> {
    let logits_v: Vec<f32> = logits.to_vec1()?;
    let next_token = logits_v
        .iter()
        .enumerate()
        .max_by(|(_, u), (_, v)| u.total_cmp(v))
        .map(|(i, _)| i as u32)
        .unwrap();
    Ok(next_token)
}

fn sample_multinomial(prs: &Vec<f32>) -> Result<u32> {
    let time = ic_cdk::api::time();
    let seed = hash_time(time);
    let next_token = deterministic_weighted_sample(&prs, seed);
    Ok(next_token as u32)
}

/// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
/// probability top_p. This way we never sample tokens that have very low probabilities and are
/// less likely to go "off the rails".
fn sample_topp(prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
    let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

    // Sort by descending probability.
    argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

    // Clamp smaller probabilities to zero.
    let mut cumsum = 0.;
    for index in &argsort_indices {
        if cumsum >= top_p {
            prs[*index] = 0.0;
        } else {
            cumsum += prs[*index];
        }
    }
    // Sample with clamped probabilities.
    sample_multinomial(prs)
}

// top-k sampling samples from the k tokens with the largest probabilities.
fn sample_topk(prs: &mut Vec<f32>, top_k: usize) -> Result<u32> {
    if top_k >= prs.len() {
        sample_multinomial(prs)
    } else {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        let (indices, _, _) =
            argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
        let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
        let index = sample_multinomial(&prs)?;
        Ok(indices[index as usize] as u32)
    }
}

// top-k sampling samples from the k tokens with the largest probabilities.
// then top-p sampling.
fn sample_topk_topp(prs: &mut Vec<f32>, top_k: usize, top_p: f32) -> Result<u32> {
    if top_k >= prs.len() {
        sample_topp(prs, top_p)
    } else {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        let (indices, _, _) =
            argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
        let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
        let sum_p = prs.iter().sum::<f32>();
        let index = if top_p <= 0.0 || top_p >= sum_p {
            sample_multinomial(&prs)?
        } else {
            sample_topp(&mut prs, top_p)?
        };
        Ok(indices[index as usize] as u32)
    }
}

pub fn sample(logits: &Tensor, temperature: f64, top_k: Option<usize>, top_p: Option<f32>) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    // Handle zero temperature case (argmax)
    if temperature == 0.0 {
        sample_argmax(logits)
    } else {
        // Calculate probabilities
        let mut prs = {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax(&logits, 0)?;
            prs.to_vec1()?
        };

        // Use match for Options
        match (top_k, top_p) {
            (Some(k), Some(p)) => sample_topk_topp(&mut prs, k, p),
            (Some(k), None) => sample_topk(&mut prs, k),
            (None, Some(p)) => {
                if p <= 0.0 || p >= 1.0 {
                    sample_multinomial(&prs)
                } else {
                    sample_topp(&mut prs, p)
                }
            },
            (None, None) => sample_multinomial(&prs),
        }
    }
}




fn hash_time(time: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    time.hash(&mut hasher);
    hasher.finish()
}


fn deterministic_weighted_sample(weights: &[f32], seed: u64) -> usize {
    let total_weight: f32 = weights.iter().sum();
    let seed_float = (seed as f64 / u64::MAX as f64) as f32;
    let target = seed_float * total_weight;

    let mut cumulative_weight = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative_weight += weight;
        if cumulative_weight > target {
            return index;
        }
    }

    // Fallback to last index (should rarely happen due to floating-point precision)
    weights.len() - 1
}

