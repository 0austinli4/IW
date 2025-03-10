import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import special


def cache_size(n_layers, d_model):
    # https://github.com/qoofyk/LLM_Sizing_Guide/
    # https://blogs.vmware.com/cloud-foundation/2024/09/25/llm-inference-sizing-and-performance-guidance/
    # define the cache size as a function of number of tokens???
    # 1 token == X size
    # kv_cache_size_per_token
    # = (2 × 2 × n_layers × d_model) bytes/token
    # = (4 × 32 × 4096) bytes/token
    # = 524288 bytes/token
    # ~ 0.00049 GiB/token for Llama-3-8b
    return (2 * 2 * n_layers * d_model) / (10**9)


def cache_capacity(n_objects, block_size, mem_per_token):
    # n_objects * 512 tokens per object * X mem per token)
    # https://huggingface.co/meta-llama/Llama-2-70b
    return n_objects * block_size * mem_per_token


def plot_request_distribution(request_data):
    print("Plotting request distribution")

    # Count occurrences of each hash_id
    distribution = Counter(item["hash_id"] for item in request_data)  # Count by hash_id
    print(distribution)

    # Sort by frequency
    sorted_distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    frequencies = [freq for _, freq in sorted_distribution]
    ranks = np.arange(1, len(frequencies) + 1)

    # Calculate statistical metrics
    mean_freq = np.mean(frequencies)
    median_freq = np.median(frequencies)
    percentiles = [np.percentile(frequencies, p) for p in [90, 95, 99]]
    
    # Gini coefficient calculation
    def gini(x):
        sorted_x = np.sort(x)
        index = np.arange(1, len(x) + 1)
        n = len(x)
        return (np.sum((2 * index - n - 1) * sorted_x)) / (n * np.sum(sorted_x))
    
    gini_coefficient = gini(frequencies)

    # Calculate Zipf law fit
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    zipf_slope, _ = np.polyfit(log_ranks, log_freqs, 1)

    # Calculate moving average for smoothing
    window_size = 5  # Size of the moving average window
    smoothed_frequencies = np.convolve(
        frequencies, np.ones(window_size) / window_size, mode="valid"
    )
    smoothed_ranks = ranks[window_size - 1 :]  # Adjust ranks for the smoothed data

    # Plotting
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Log-Log Frequency-Rank Curve
    plt.subplot(2, 2, 1)
    plt.loglog(
        ranks, frequencies, marker="o", linestyle="-", alpha=0.7, label="Original"
    )
    plt.loglog(
        smoothed_ranks,
        smoothed_frequencies,
        marker="x",
        linestyle="-",
        color="red",
        alpha=0.7,
        label="Smoothed",
    )
    plt.xlabel("Rank of Object", fontsize=12)
    plt.ylabel("Frequency of Requests", fontsize=12)
    plt.title("Frequency-Rank Distribution (Log-Log)", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Subplot 2: Cumulative Distribution
    plt.subplot(2, 2, 2)
    cumulative_freqs = np.cumsum(sorted(frequencies, reverse=True)) / sum(frequencies)
    plt.plot(ranks, cumulative_freqs, marker='o')
    plt.xscale('log')
    plt.xlabel("Rank of Object", fontsize=12)
    plt.ylabel("Cumulative Fraction of Requests", fontsize=12)
    plt.title("Cumulative Request Distribution", fontsize=14)
    plt.grid(True)

    # Subplot 3: Histogram of Frequencies
    plt.subplot(2, 2, 3)
    plt.hist(frequencies, bins='auto', alpha=0.7)
    plt.xlabel("Request Frequency", fontsize=12)
    plt.ylabel("Number of Objects", fontsize=12)
    plt.title("Histogram of Request Frequencies", fontsize=14)

    # Subplot 4: Top N Objects Pie Chart
    plt.subplot(2, 2, 4)
    top_n = 10
    top_objects = sorted_distribution[:top_n]
    plt.pie([freq for _, freq in top_objects], 
            labels=[str(hash_id)[:10] for hash_id, _ in top_objects],
            autopct='%1.1f%%')
    plt.title(f"Top {top_n} Most Requested Objects", fontsize=14)

    plt.tight_layout()
    plt.show()

    # Print statistical insights
    print(f"Total unique objects: {len(distribution)}")
    print(f"Mean request frequency: {mean_freq:.2f}")
    print(f"Median request frequency: {median_freq:.2f}")
    print(f"90th percentile frequency: {percentiles[0]:.2f}")
    print(f"95th percentile frequency: {percentiles[1]:.2f}")
    print(f"99th percentile frequency: {percentiles[2]:.2f}")
    print(f"Gini coefficient: {gini_coefficient:.4f}")
    print(f"Zipf law slope: {zipf_slope:.4f}")

    return {
        'total_objects': len(distribution),
        'mean_frequency': mean_freq,
        'median_frequency': median_freq,
        'percentiles': percentiles,
        'gini_coefficient': gini_coefficient,
        'zipf_slope': zipf_slope
    }


def extract_hash_ids(input_file, output_file):
    """
    Extract hash_ids from a JSONL file and write them to a text file.

    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output text file
    """
    with open(input_file, "r") as jsonl_file, open(output_file, "w") as txt_file:
        for line in jsonl_file:
            try:
                # Parse each line as a JSON object
                data = json.loads(line.strip())

                # Extract hash_id and write to text file
                hash_id = data.get("hash_ids")

                for element in hash_id:
                    txt_file.write(f"{element}\n")

                # if hash_id is not None:
                #     txt_file.write(f"{hash_id}\n")
            except json.JSONDecodeError:
                # Skip lines that can't be parsed
                raise Exception("JSON can't be decoded")


def get_data(json_file_path):
    """
    Read and parse JSON Lines (JSONL) file, returning request data.

    Args:
        json_file_path (str): Path to the JSONL file containing request data.

    Returns:
        list: List of dictionaries containing timestamp and hash_id.
    """
    request_data = []
    with open(json_file_path, "r") as f:
        for line in f:
            try:
                request = json.loads(line.strip())
                timestamp = request.get("timestamp")
                hash_ids = request.get("hash_ids", [])
                for hash_id in hash_ids:
                    request_data.append({"timestamp": timestamp, "hash_id": hash_id})
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line}")

    # Gather frequencies using Counter
    frequency_distribution = Counter(item["hash_id"] for item in request_data)
    # print(frequency_distribution)
    one_hit_wonders = sum(1 for count in frequency_distribution.values() if count == 1)

    print("One hit wonders", (one_hit_wonders))
    print("Total objects", len(frequency_distribution))
    print("Ratio of one-hit wonders", one_hit_wonders / len(frequency_distribution))

    return request_data


def test_zipf_law(frequencies):
    """
    Test if the frequency distribution follows Zipf's law.

    Args:
        frequencies (list or np.array): Frequencies of items

    Returns:
        dict: Statistical analysis of Zipf's law adherence
    """
    # Sort frequencies in descending order
    sorted_freq = np.sort(frequencies)[::-1]

    # Rank of each frequency (1-indexed)
    ranks = np.arange(1, len(sorted_freq) + 1)

    # Theoretical Zipf distribution
    zipf_dist = 1 / (ranks**1)
    zipf_dist /= zipf_dist.sum()

    # Normalize observed frequencies
    observed_freq = sorted_freq / sorted_freq.sum()

    # Statistical tests
    # 1. Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_1samp(observed_freq, zipf_dist)

    # 2. Log-log regression to check power law
    log_ranks = np.log(ranks)
    log_freq = np.log(observed_freq)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_freq)

    return {
        "ks_statistic": ks_statistic,
        "ks_pvalue": ks_pvalue,
        "zipf_exponent": -slope,  # Negative slope of log-log regression
        "r_squared": r_value**2,
        "p_value": p_value,
    }


def plot_top_popular_objects(popular_objects):
    """
    Plot a bar chart of the top popular objects.

    Args:
        popular_objects (list): List of tuples (hash_id, frequency)
    """
    # Extracting hash IDs and frequencies
    objects = [str(obj[0]) for obj in popular_objects]  # Ensure hash IDs are strings
    frequencies = [obj[1] for obj in popular_objects]
    plt.figure(figsize=(10, 6))
    plt.bar(objects, frequencies, color="blue")
    plt.title("Top 20 Popular Objects")
    plt.xlabel("Hash ID")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_inter_arrival_times(inter_arrival_times):
    """
    Plot histograms of inter-arrival times for the top objects with improved readability.

    Args:
        inter_arrival_times (dict): Dictionary of inter-arrival times for each object.
    """
    plt.figure(figsize=(12, 6))

    num_hashes = len(inter_arrival_times)
    max_bins = 20  # Reduce the number of bins for clarity

    for hash_id, times in inter_arrival_times.items():
        if len(times) > 0:
            plt.hist(
                times,
                bins=min(max_bins, len(times)),
                alpha=0.6,
                label=str(hash_id),
                edgecolor="black",
                linewidth=0.5,
            )

    plt.title("Inter-arrival Time Distribution")
    plt.xlabel("Inter-arrival Time (seconds)")
    plt.ylabel("Frequency")
    plt.yscale("log")  # Use log scale if there's a large range in values
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    if num_hashes <= 10:  # Show legend only if not too many labels
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_spatial_locality(spatial_locality):
    """
    Plot scatter plot for spatial locality analysis.

    Args:
        spatial_locality (dict): Dictionary of spatial locality data
    """
    plt.figure(figsize=(10, 6))
    x_values = []
    y_values = []
    for current_id, next_ids in spatial_locality.items():
        for next_id in next_ids:
            x_values.append(current_id)
            y_values.append(next_id)
    plt.scatter(x_values, y_values, alpha=0.5)
    plt.title("Spatial Locality Analysis")
    plt.xlabel("Current Hash ID")
    plt.ylabel("Next Hash ID")
    plt.tight_layout()
    plt.show()


def plot_one_hit_wonders(one_hit_wonders, total_unique):
    """
    Plot a pie chart for one-hit wonders.

    Args:
        one_hit_wonders (float): Proportion of one-hit wonders
        total_unique (int): Total number of unique objects
    """
    labels = ["One-hit Wonders", "Other Objects"]
    sizes = [one_hit_wonders * total_unique, (1 - one_hit_wonders) * total_unique]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("One-hit Wonders Proportion")
    plt.axis("equal")  # Equal aspect ratio ensures pie chart is circular.
    plt.show()


def plot_zipf_distribution(request_data):
    """
    Plot the Zipf distribution based on request data.

    Args:
        request_data (list): List of dictionaries containing timestamp and hash_id.
    """
    # Count occurrences of each hash_id
    distribution = Counter(item["hash_id"] for item in request_data)  # Count by hash_id

    # Limit to the top 50 most common hash IDs
    top_n = 100
    most_common = distribution.most_common(top_n)
    frequencies = [freq for _, freq in most_common]

    # Calculate Zipf distribution
    a = 2.0  # Distribution parameter
    x = np.arange(1, top_n + 1)
    zipf_distribution = x ** (-a) / special.zetac(a)

    # Plot
    plt.figure(figsize=(14, 8))
    plt.bar(x, frequencies, alpha=0.7, label="Observed Frequencies")
    plt.plot(
        x,
        zipf_distribution * max(frequencies),
        color="red",
        linewidth=2,
        label="Zipf Distribution",
    )
    plt.xlabel("Rank of Object", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Zipf Distribution of Object Requests", fontsize=16)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def main():
    json_file_path = "sub_trace.txt"

    # plot_request_distribution(json_file_path)
    analysis = get_data(json_file_path)
    plot_request_distribution(analysis)
    plot_zipf_distribution(analysis)


if __name__ == "__main__":
    main()
