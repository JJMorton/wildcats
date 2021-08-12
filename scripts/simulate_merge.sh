#!/bin/bash

#SBATCH --job-name=slim_merge
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:30
#SBATCH --mem=100M
#SBATCH --output ../output/slurm/slurm-%j.out

header="index,divergence_domestic_captive,divergence_domestic_wild,divergence_wild_captive,diversity_all_pops,diversity_captive,diversity_domestic,diversity_wild,expected_heterozygosity_all_pops,expected_heterozygosity_captive,expected_heterozygosity_domestic,expected_heterozygosity_wild,f2_domestic_captive,f2_domestic_wild,f2_wild_captive,f3_captive,f3_domestic,f3_wild,fst_domestic_captive,fst_domestic_wild,fst_wild_captive,monomorphic_sites_captive,monomorphic_sites_domestic,monomorphic_sites_wild,observed_heterozygosity_all_pops,observed_heterozygosity_captive,observed_heterozygosity_domestic,observed_heterozygosity_wild,pc1_iqr_all_pops,pc1_iqr_captive,pc1_iqr_domestic,pc1_iqr_wild,pc1_median_all_pops,pc1_median_captive,pc1_median_dist_domestic_captive,pc1_median_dist_domestic_wild,pc1_median_dist_wild_captive,pc1_median_domestic,pc1_median_wild,pc2_iqr_all_pops,pc2_iqr_captive,pc2_iqr_domestic,pc2_iqr_wild,pc2_median_all_pops,pc2_median_captive,pc2_median_dist_domestic_captive,pc2_median_dist_domestic_wild,pc2_median_dist_wild_captive,pc2_median_domestic,pc2_median_wild,r2_all_pops_0.5_1e6,r2_all_pops_0_0.5mb,r2_all_pops_1_2mb,r2_all_pops_2_4mb,r2_captive_0.5_1e6,r2_captive_0_0.5mb,r2_captive_1_2mb,r2_captive_2_4mb,r2_domestic_0.5_1e6,r2_domestic_0_0.5mb,r2_domestic_1_2mb,r2_domestic_2_4mb,r2_wild_0.5_1e6,r2_wild_0_0.5mb,r2_wild_1_2mb,r2_wild_2_4mb,roh_all_pops_iqr,roh_all_pops_mean,roh_captive_iqr,roh_captive_mean,roh_domestic_iqr,roh_domestic_mean,roh_wild_iqr,roh_wild_mean,segregating_sites,sfs_mean_all_pops_0_9,sfs_mean_all_pops_18_27,sfs_mean_all_pops_27_36,sfs_mean_all_pops_36_45,sfs_mean_all_pops_9_18,sfs_mean_captive_0_2,sfs_mean_captive_2_4,sfs_mean_captive_4_6,sfs_mean_captive_6_8,sfs_mean_captive_8_10,sfs_mean_domestic_0_1,sfs_mean_domestic_1_2,sfs_mean_domestic_2_3,sfs_mean_domestic_3_4,sfs_mean_domestic_4_5,sfs_mean_wild_0_6,sfs_mean_wild_12_18,sfs_mean_wild_18_24,sfs_mean_wild_24_30,sfs_mean_wild_6_12,tajimas_d_all_pops,tajimas_d_captive,tajimas_d_domestic,tajimas_d_wild,wattersons_theta_all_pops,wattersons_theta_captive,wattersons_theta_domestic,wattersons_theta_wild,random_seed"
echo "$header" > ../output/stats.csv
for file in ../output/stats/stats_*.csv; do
	echo -ne "\r                                                          "
	echo -ne "\rMerging file $file"
	i=$(echo $file | grep -o '[0-9]*.csv' | grep -o '[0-9]*')
	echo -n "$i," >> ../output/stats.csv
	cat $file >> ../output/stats.csv
done
echo

