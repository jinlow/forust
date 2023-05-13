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
    fn sample(&mut self, rng: &mut StdRng, index: &[usize]) -> (Vec<usize>, Vec<usize>);
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
    fn sample(&mut self, rng: &mut StdRng, index: &[usize]) -> (Vec<usize>, Vec<usize>) {
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
pub struct GossSampler<'a> {
    gradient: Option<&'a [f64]>,
}

impl<'a> Default for GossSampler<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl<'a> GossSampler<'a> {
    pub fn new() -> Self {
        GossSampler { gradient: None }
    }
    pub fn add_gradient(&mut self, gradient: &'a [f64]) {
        self.gradient = Some(gradient);
    }
}

impl<'a> Sampler for GossSampler<'a> {
    #[allow(unused_variables)]
    fn sample(&mut self, rng: &mut StdRng, index: &[usize]) -> (Vec<usize>, Vec<usize>) {
        todo!()
    }
}
