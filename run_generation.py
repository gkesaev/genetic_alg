#!/usr/bin/env python3
"""
Run genetic algorithm to approximate Mona Lisa using circles.
Saves output images instead of displaying them.
"""
import os
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, ceil
from tqdm import tqdm

from dna import DNA
from genes import Circle
from generation import Generation


def save_population_snapshot(gen: Generation, iter_num: int, output_dir='output'):
    """Save population snapshot to file."""
    os.makedirs(output_dir, exist_ok=True)

    num_plots_per_axis = ceil(sqrt(len(gen.generation)))
    fig, axs = plt.subplots(num_plots_per_axis, num_plots_per_axis,
                            figsize=(15, 15),
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.3})
    fig.suptitle(f'Generation {iter_num} - Population Sample', fontsize=16)

    gen_idx = 0
    try:
        for i in range(num_plots_per_axis):
            for j in range(num_plots_per_axis):
                if num_plots_per_axis == 1:
                    ax = axs
                elif len(axs.shape) == 1:
                    ax = axs[max(i, j)]
                else:
                    ax = axs[i, j]

                ax.set_title(f'DNA {gen_idx}\nFitness: {gen.generation[gen_idx].fitness_cost:.2f}',
                            fontsize=10)
                result_img = gen.generation[gen_idx].grow_result()
                ax.imshow(np.clip(result_img, 0, 255).astype(np.uint8))
                ax.imshow(gen.image_to_esimtate, alpha=0.4)
                ax.axis('off')
                gen_idx += 1
    except IndexError:
        if num_plots_per_axis > 1:
            for i in range(gen_idx, num_plots_per_axis * num_plots_per_axis):
                row = i // num_plots_per_axis
                col = i % num_plots_per_axis
                if len(axs.shape) == 1:
                    axs[max(row, col)].axis('off')
                else:
                    axs[row, col].axis('off')

    filename = f'{output_dir}/generation_{iter_num:04d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved snapshot: {filename}')


def run_with_snapshots(gen: Generation, output_dir='output'):
    """Run genetic algorithm with periodic snapshots."""
    os.makedirs(output_dir, exist_ok=True)

    # Save initial population
    save_population_snapshot(gen, 0, output_dir)

    # Run iterations
    for i in tqdm(range(gen.num_iterations), desc='Running generations'):
        n_best = gen.get_n_best(4)
        gen.new_generation(n_best)

        # Save snapshot every 10 iterations and at the end
        if i % 10 == 0 or i == gen.num_iterations - 1:
            save_population_snapshot(gen, i + 1, output_dir)

            # Save best individual
            best = gen.get_n_best(1)[0][1]
            best_img = best.grow_result()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # Best result alone
            ax1.imshow(np.clip(best_img, 0, 255).astype(np.uint8))
            ax1.set_title(f'Best Result (Gen {i+1})\nFitness: {best.fitness_cost:.2f}')
            ax1.axis('off')

            # Overlay with original
            ax2.imshow(np.clip(best_img, 0, 255).astype(np.uint8))
            ax2.imshow(gen.image_to_esimtate, alpha=0.5)
            ax2.set_title('Overlay with Original')
            ax2.axis('off')

            # Original
            ax3.imshow(gen.image_to_esimtate)
            ax3.set_title('Original Image')
            ax3.axis('off')

            filename = f'{output_dir}/best_gen_{i+1:04d}.png'
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved best: {filename}')

    # Final results
    save_population_snapshot(gen, gen.num_iterations, output_dir)
    best = gen.get_n_best(1)[0][1]

    return best


if __name__ == '__main__':
    print("Loading Mona Lisa image...")
    img = imageio.imread('mona-lisa.jpg!HalfHD.jpg')
    print(f"Image shape: {img.shape}")

    # Configuration
    config = {
        'population_count': 9,      # 9 DNAs per generation
        'num_iter': 300,            # 300 generations
        'genes_per_dna': 40,        # 40 circles per image
    }

    print(f"\nGenetic Algorithm Configuration:")
    print(f"  Population size: {config['population_count']}")
    print(f"  Generations: {config['num_iter']}")
    print(f"  Circles per image: {config['genes_per_dna']}")
    print(f"  Total iterations: {config['population_count'] * config['num_iter']}")

    print("\nCreating initial generation...")
    gen = Generation(
        population_count=config['population_count'],
        num_iter=config['num_iter'],
        species_kind=Circle,
        image_to_esimtate=img,
        genes_per_dna=config['genes_per_dna']
    )

    print("\nRunning genetic algorithm...")
    output_dir = 'output'
    best_dna = run_with_snapshots(gen, output_dir)

    # Save final best result
    final_img = best_dna.grow_result()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(np.clip(final_img, 0, 255).astype(np.uint8))
    axes[0].set_title(f'Final Result\n{len(best_dna)} circles, Fitness: {best_dna.fitness_cost:.2f}')
    axes[0].axis('off')

    axes[1].imshow(np.clip(final_img, 0, 255).astype(np.uint8))
    axes[1].imshow(img, alpha=0.5)
    axes[1].set_title('Overlay with Original')
    axes[1].axis('off')

    axes[2].imshow(img)
    axes[2].set_title('Original Image')
    axes[2].axis('off')

    final_filename = f'{output_dir}/final_result.png'
    plt.savefig(final_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also save just the result image
    imageio.imwrite(f'{output_dir}/final_result_only.png',
                    np.clip(final_img, 0, 255).astype(np.uint8))

    print(f"\n{'='*60}")
    print(f"COMPLETED!")
    print(f"{'='*60}")
    print(f"Final DNA has {len(best_dna)} circles")
    print(f"Final fitness: {best_dna.fitness_cost:.2f}")
    print(f"Results saved to '{output_dir}/' directory")
    print(f"Final comparison: {final_filename}")
    print(f"Final result only: {output_dir}/final_result_only.png")
