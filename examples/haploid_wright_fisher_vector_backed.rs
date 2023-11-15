use ancestry_graph::vector_backed_graph::*;

fn haploid_wf(popsize: usize, ngenerations: i64, genome_length: i64, seed: u64) -> Graph {
    use rand::Rng;
    use rand::SeedableRng;

    let mut graph = Graph::new(genome_length);
    let mut parents = vec![];
    for _ in 0..popsize {
        parents.push(graph.add_birth())
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sample_parent = rand::distributions::Uniform::new(0, popsize);
    let sample_breakpoint = rand::distributions::Uniform::new(1, genome_length);
    let mut children = vec![];

    for gen in 0..ngenerations {
        graph.current_time += 1;
        for _ in 0..popsize {
            //for &i in &parents {
            //    // mark them as "dead".
            //    graph.enqueue_parent(i);
            //}
            let child = graph.add_birth();
            children.push(child);
            let pindex = rng.sample(sample_parent);
            let left_parent = parents[pindex];
            let pindex = rng.sample(sample_parent);
            let right_parent = parents[pindex];
            let breakpoint = rng.sample(sample_breakpoint);
            //graph.enqueue_parent(left_parent);
            //graph.enqueue_parent(right_parent);
            graph.record_transmission(0, breakpoint, left_parent, child);
            graph.record_transmission(breakpoint, genome_length, right_parent, child);
        }
        propagate_changes(&parents, &mut graph);

        std::mem::swap(&mut parents, &mut children);
        children.clear();
    }

    graph
}

pub fn main() {
    let graph = haploid_wf(1000, 5000, 10000000, 213512);
    let num_nodes = graph.tables.nodes.birth_time.iter().cloned().filter(|&t| t != -1).count();
    let mut num_edges = 0;
    for e in graph.tables.edges.iter().filter(|e|!e.is_empty()) {
        num_edges += e.len()
    }
    println!("{num_nodes} {num_edges}")
}
