use rand::rngs::StdRng;
use rand::Rng;

// A sampler can be used to subset the data prior to fitting a new tree.
pub trait Sampler {
    /// Sample the data, returning a tuple, where the first item is the samples
    /// chosen for training, and the second are the samples excluded.
    fn sample(&mut self, index: &[usize]) -> (Vec<usize>, Vec<usize>);
}

pub struct RandomSampler<'a> {
    subsample: f32,
    rng: &'a mut StdRng,
}

impl<'a> RandomSampler<'a> {
    #[allow(dead_code)]
    pub fn new(rng: &'a mut StdRng, subsample: f32) -> Self {
        RandomSampler { subsample, rng }
    }
}

impl<'a> Sampler for RandomSampler<'a> {
    fn sample(&mut self, index: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let subsample = self.subsample;
        let mut chosen = Vec::new();
        let mut excluded = Vec::new();
        for i in index {
            if self.rng.gen_range(0.0..1.0) < subsample {
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
    rng: &'a mut StdRng,
    gradient: Option<&'a [f64]>,
}

impl<'a> GossSampler<'a> {
    #[allow(dead_code)]
    pub fn new(rng: &'a mut StdRng) -> Self {
        GossSampler {
            rng,
            gradient: None,
        }
    }
}
