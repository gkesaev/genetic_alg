#!/usr/bin/env python3
"""
Improved genetic algorithm with transparency and smaller circles for better detail.
"""
import os
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, ceil
from tqdm import tqdm
from skimage import draw


class ImprovedCircle:
    """Circle with alpha transparency for better layering."""

    def __init__(self, r, g, b, x, y, radius, alpha=128):
        self.r = np.clip(r, 0, 255)
        self.g = np.clip(g, 0, 255)
        self.b = np.clip(b, 0, 255)
        self.x = int(x)
        self.y = int(y)
        self.radius = max(1, int(radius))
        self.alpha = np.clip(alpha, 0, 255) / 255.0  # Normalize to 0-1

    def draw_on(self, arr: np.array) -> np.array:
        """Draw circle with alpha blending."""
        try:
            rr, cc = draw.disk(center=(self.y, self.x),
                              radius=self.radius,
                              shape=arr.shape[:2])

            # Alpha blending: new_color = alpha * circle_color + (1 - alpha) * background
            arr[rr, cc, 0] = self.alpha * self.r + (1 - self.alpha) * arr[rr, cc, 0]
            arr[rr, cc, 1] = self.alpha * self.g + (1 - self.alpha) * arr[rr, cc, 1]
            arr[rr, cc, 2] = self.alpha * self.b + (1 - self.alpha) * arr[rr, cc, 2]
        except Exception:
            pass  # Skip circles that are out of bounds

        return arr

    @staticmethod
    def generate_random(img_shape, min_radius=3, max_radius=30):
        """Generate random circle with proper parameters."""
        h, w = img_shape[:2]

        # Position anywhere in image (with small margin)
        x = random.randint(-20, w + 20)
        y = random.randint(-20, h + 20)

        # Smaller, more detailed circles
        radius = random.randint(min_radius, max_radius)

        # Independent RGB channels (0-255 each)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Alpha for transparency (higher = more opaque)
        alpha = random.randint(30, 150)

        return np.array([r, g, b, x, y, radius, alpha])

    @staticmethod
    def mutate(arr: np.ndarray, img_shape):
        """Mutate circle parameters."""
        h, w = img_shape[:2]

        # r, g, b, x, y, radius, alpha
        r = int(np.clip(arr[0] + random.randint(-20, 20), 0, 255))
        g = int(np.clip(arr[1] + random.randint(-20, 20), 0, 255))
        b = int(np.clip(arr[2] + random.randint(-20, 20), 0, 255))

        x = arr[3] + random.randint(-10, 10)
        y = arr[4] + random.randint(-10, 10)

        radius = max(1, int(arr[5] + random.randint(-3, 3)))
        alpha = int(np.clip(arr[6] + random.randint(-20, 20), 10, 200))

        return np.array([r, g, b, x, y, radius, alpha])


class ImprovedDNA:
    """DNA with transparency-enabled circles."""

    def __init__(self, target_image: np.ndarray, num_circles: int = 100):
        self.target_image = target_image
        self.shape = target_image.shape
        self.num_circles = num_circles

        # Generate random circles
        self.genes = np.array([
            ImprovedCircle.generate_random(self.shape)
            for _ in range(num_circles)
        ])

        self.fitness_cost = self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate fitness as color distance from target."""
        result = self.render()
        return np.sum(np.abs(result.astype(np.float32) - self.target_image.astype(np.float32)))

    def render(self):
        """Render the DNA as an image."""
        result = np.zeros_like(self.target_image, dtype=np.float32)

        for gene in self.genes:
            circle = ImprovedCircle(*gene)
            result = circle.draw_on(result)

        return np.clip(result, 0, 255).astype(np.uint8)

    def mutate(self, mutation_rate=0.3):
        """Create a mutated copy of this DNA."""
        new_genes = self.genes.copy()

        # Mutate some percentage of genes
        num_mutations = max(1, int(len(new_genes) * mutation_rate))
        indices = random.sample(range(len(new_genes)), num_mutations)

        for idx in indices:
            new_genes[idx] = ImprovedCircle.mutate(new_genes[idx], self.shape)

        # Occasionally add or remove a circle
        if random.random() < 0.05 and len(new_genes) < 300:
            # Add a circle
            new_gene = ImprovedCircle.generate_random(self.shape)
            new_genes = np.vstack([new_genes, new_gene])
        elif random.random() < 0.02 and len(new_genes) > 50:
            # Remove a random circle
            idx = random.randint(0, len(new_genes) - 1)
            new_genes = np.delete(new_genes, idx, axis=0)

        new_dna = ImprovedDNA.__new__(ImprovedDNA)
        new_dna.target_image = self.target_image
        new_dna.shape = self.shape
        new_dna.genes = new_genes
        new_dna.num_circles = len(new_genes)
        new_dna.fitness_cost = new_dna.calculate_fitness()

        return new_dna


def run_improved_algorithm(target_image, num_iterations=1000, population_size=10,
                          initial_circles=150, output_dir='output_improved'):
    """Run improved genetic algorithm."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"Initializing population of {population_size} DNAs with {initial_circles} circles each...")
    population = [ImprovedDNA(target_image, initial_circles) for _ in range(population_size)]

    # Track best
    best_dna = min(population, key=lambda d: d.fitness_cost)
    best_fitness_history = [best_dna.fitness_cost]

    print(f"Initial best fitness: {best_dna.fitness_cost:.2f}")

    # Save initial
    save_comparison(best_dna, target_image, 0, output_dir)

    for iteration in tqdm(range(num_iterations), desc="Evolution"):
        # Sort by fitness
        population.sort(key=lambda d: d.fitness_cost)

        # Keep top 30%
        num_survivors = max(2, population_size // 3)
        survivors = population[:num_survivors]

        # Generate new population
        new_population = survivors.copy()

        while len(new_population) < population_size:
            # Select parent from survivors (bias toward better fitness)
            parent = random.choices(survivors, weights=[1/(i+1) for i in range(len(survivors))])[0]
            child = parent.mutate(mutation_rate=0.2)
            new_population.append(child)

        population = new_population

        # Track best
        current_best = min(population, key=lambda d: d.fitness_cost)
        if current_best.fitness_cost < best_dna.fitness_cost:
            best_dna = current_best

        best_fitness_history.append(best_dna.fitness_cost)

        # Save progress
        if (iteration + 1) % 50 == 0:
            save_comparison(best_dna, target_image, iteration + 1, output_dir)
            print(f"\nIteration {iteration + 1}: Best fitness = {best_dna.fitness_cost:.2f}, "
                  f"Circles = {best_dna.num_circles}")

    # Save final result
    save_comparison(best_dna, target_image, num_iterations, output_dir, final=True)

    # Save fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (lower is better)')
    plt.title('Fitness Evolution Over Time')
    plt.grid(True)
    plt.savefig(f'{output_dir}/fitness_history.png', dpi=100, bbox_inches='tight')
    plt.close()

    return best_dna


def save_comparison(dna, target, iteration, output_dir, final=False):
    """Save comparison image."""
    result = dna.render()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(result)
    axes[0].set_title(f'Generated ({dna.num_circles} circles)\nFitness: {dna.fitness_cost:.0f}')
    axes[0].axis('off')

    axes[1].imshow(result)
    axes[1].imshow(target, alpha=0.5)
    axes[1].set_title('Overlay with Original')
    axes[1].axis('off')

    axes[2].imshow(target)
    axes[2].set_title('Original Image')
    axes[2].axis('off')

    fig.suptitle(f'Iteration {iteration}', fontsize=16)

    filename = f'{output_dir}/{"final" if final else f"iter_{iteration:04d}"}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    # Also save just the result
    imageio.imwrite(f'{output_dir}/{"final_result" if final else f"result_{iteration:04d}"}.png', result)


if __name__ == '__main__':
    print("Loading Mona Lisa image...")
    img = imageio.imread('mona-lisa.jpg!HalfHD.jpg')
    print(f"Image shape: {img.shape}")

    print("\nImproved Configuration:")
    print("  - Initial circles: 150 (with transparency)")
    print("  - Circle radius: 3-30 pixels (much smaller for detail)")
    print("  - Alpha transparency: 30-150 (allows layering)")
    print("  - Population: 10 DNAs")
    print("  - Iterations: 1000")
    print("  - Dynamic circle count (can add/remove circles)")
    print()

    best = run_improved_algorithm(
        img,
        num_iterations=1000,
        population_size=10,
        initial_circles=150,
        output_dir='output_improved'
    )

    print(f"\n{'='*60}")
    print("COMPLETED!")
    print(f"{'='*60}")
    print(f"Final DNA: {best.num_circles} circles")
    print(f"Final fitness: {best.fitness_cost:.2f}")
    print(f"Results saved to 'output_improved/' directory")
