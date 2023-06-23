use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum SampleMethod {
    None,
    Random,
    Goss,
}

// A sampler can be used to subset the data prior to fitting a new tree.
pub trait Sampler {
    /// Sample the data, returning a tuple, where the first item is the samples
    /// chosen for training, and the second are the samples excluded.
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>);
}

pub struct RandomSampler {
    subsample: f32,
}

impl RandomSampler {
    #[allow(dead_code)]
    pub fn new(subsample: f32) -> Self {
        RandomSampler { subsample }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        _grad: &mut [f32],
        _hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>) {
        let subsample = self.subsample;
        let mut chosen = Vec::new();
        let mut excluded = Vec::new();
        for i in index {
            if rng.gen_range(0.0..1.0) < subsample {
                chosen.push(*i);
            } else {
                excluded.push(*i)
            }
        }
        (chosen, excluded)
    }
}

#[allow(dead_code)]
pub struct GossSampler {
    a: f64, // https://lightgbm.readthedocs.io/en/latest/Parameters.html#top_rate
    b: f64, // https://lightgbm.readthedocs.io/en/latest/Parameters.html#other_rate
}

impl Default for GossSampler {
    fn default() -> Self {
        GossSampler { a: 0.2, b: 0.1 }
    }
}

#[allow(dead_code)]
impl GossSampler {
    pub fn new(a: f64, b: f64) -> Self {
        GossSampler { a, b }
    }
}

impl Sampler for GossSampler {
    fn sample(
        &mut self,
        rng: &mut StdRng,
        index: &[usize],
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<usize>, Vec<usize>) {
        let fact = ((1. - self.a) / self.b) as f32;
        let top_n = (self.a * index.len() as f64) as usize;
        let rand_n = (self.b * index.len() as f64) as usize;

        // sort gradient by absolute value from highest to lowest
        let mut sorted = (0..index.len()).collect::<Vec<_>>();
        sorted.sort_unstable_by(|&a, &b| grad[b].abs().total_cmp(&grad[a].abs()));

        // select the topN largest gradients
        let mut used_set = sorted[0..top_n].to_vec();

        // sample the rest based on randN
        let subsample = rand_n as f64 / (index.len() as f64 - top_n as f64);

        // weight the sampled "small gradients" by fact and append indices to used_set
        for i in &sorted[top_n..sorted.len()] {
            if rng.gen_range(0.0..1.0) < subsample {
                grad[*i] *= fact;
                hess[*i] *= fact;
                used_set.push(*i);
            }
        }

        (used_set, Vec::new())
    }
}
