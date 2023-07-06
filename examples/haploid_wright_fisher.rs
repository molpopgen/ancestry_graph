use rand::Rng;
use rand::SeedableRng;

use ancestry_graph::Graph;
use ancestry_graph::Node;

fn initialize_sim(initial_popsize: usize, genome_length: i64) -> (Graph, Vec<Node>) {
    todo!()
}

// Standard WF with exactly 1 crossover per bith.
// This is the same model as the tskit-c/-rust example.
fn simulate(seed: u64, popsize: usize, genome_length: i64, num_generations: i64) -> Graph {
    todo!()
}

fn main() {}
