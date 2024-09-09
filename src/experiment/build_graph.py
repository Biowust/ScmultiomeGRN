import gc
import os
import pandas as pd
from collections import defaultdict

from src.utils import *

def summary(data_dir):
    ans = []
    for cell_type in os.listdir(data_dir):
        atac_dir = os.path.join(data_dir, cell_type, 'atac')
        scrna_dir = os.path.join(data_dir, cell_type, 'scrna')
        if os.path.exists(atac_dir):
            barcodes = pd.read_csv(os.path.join(atac_dir, "barcodes.tsv"), header=None)
            peaks = pd.read_csv(os.path.join(atac_dir, "peaks.tsv"), header=None)
            tmp = {"cell_type": cell_type,
                  "data_type": 'ATAC',
                  "cell_num": len(barcodes),
                  "feature_num": len(peaks)}
            ans.append(tmp)
        if os.path.exists(scrna_dir):
            barcodes = pd.read_csv(os.path.join(scrna_dir, "barcodes.tsv"), header=None)
            genes = pd.read_csv(os.path.join(scrna_dir, "genes.tsv"), header=None)
            tmp = {"cell_type": cell_type,
                  "data_type": 'scRNA',
                  "cell_num": len(barcodes),
                  "feature_num": len(genes)}
            ans.append(tmp)
    return pd.DataFrame(ans)


if __name__=="__main__":


    # motif_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/human_HOCOMO"
    #
    # motif_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/HOCOMOCOv11"
    # genome_file = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/download_genome/hg19/hg19.fa"
    # promoter_file = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/gencode.v19.ProteinCoding_gene_promoter.txt"
    # genome_file = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/download_genome/hg38.fa"
    # promoter_file = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/gencode.v38.ProteinCoding_gene_promoter.txt"

    root_dirs = {
                 #"pbmc": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC/pbmc_sep_data", "19", "HOCOMOCOv11"],
                 #"lung": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/Lung_GSM4508936/lung_sep_data", "19", "HOCOMOCOv11"],
                 #"ad": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_GSE174367/ad_sep_data", "38", "HOCOMOCOv11"],
                 #"pbmc_signac": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC_signac/pbmc_signac_sep_data", "19", "HOCOMOCOv11"],
                 # "earlyAD2":["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/earlyAD/sep_data2/80", "38", "HOCOMOCOv11"],
                 # "earlyAD": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/earlyAD/sep_data", "38", "HOCOMOCOv11"],
                 # "lateAD": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/lateAD/sep_data", "38", "HOCOMOCOv11"],
                 "nonAD": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/nonAD/sep_data", "38", "HOCOMOCOv11"]
                # "early_lateAD": ["/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/early_lateAD/sep_data", "38", "HOCOMOCOv11"]

    }

    source_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/data_resource/"
    ans = []
    for key, (root_dir, hg_version, motif_version) in root_dirs.items():
        df = summary(root_dir)
        df['root_dir'] = root_dir
        df['hg_version'] = hg_version
        df['motif_version'] = motif_version
        df['dataset'] = key
        ans.append(df)
    ans = pd.concat(ans)

    for idx, group in ans.groupby("cell_type"):
        print(idx)
       # if idx=="Oligo":
       #     print("skip Oligo")
       #     continue
        row = group.iloc[0]
        # if row['cell_num']>40000:
        #     continue
        # continue
        cell_type = row['cell_type'].replace(" ", "_")
        atac_file = os.path.join(row['root_dir'], row['cell_type'], 'atac', 'matrix.mtx')
        scrna_file = os.path.join(row['root_dir'], row['cell_type'], 'scrna', 'matrix.mtx')
        genome_file = os.path.join(source_dir, "download_genome", f"hg{row['hg_version']}", f"hg{row['hg_version']}.fa")
        promoter_file = os.path.join(source_dir, f"gencode.v{row['hg_version']}.ProteinCoding_gene_promoter.txt")
        motif_dir = os.path.join(source_dir, f"{row['motif_version']}")
        save_dir = os.path.join(os.path.dirname(row['root_dir']), "graph", f"hg{row['hg_version']}_{row['motif_version']}", cell_type)

        atac_region_file = get_TFBS_from_ATAC(atac_file, motif_dir, genome_file, overwrite=False, save_dir=save_dir)

        promoter_region_file = get_TFBS_from_promoter(promoter_file, motif_dir, overwrite=False, save_dir=save_dir)

        graph_file, node_file = build_Graph(atac_region_file, promoter_region_file, overwrite=False, save_dir=save_dir)


        #cell_num = 50
        cell_num = None

        seed = 666
        if cell_num is not None:
            new_save_dir = os.path.join(os.path.dirname(row['root_dir']), f"graph_cell_{cell_num}_seed_{seed}",
                                        f"hg{row['hg_version']}_{row['motif_version']}", cell_type)
        else:
            new_save_dir = save_dir
        atac_feature_file = extract_atac_feature(atac_file, node_file, cell_num=cell_num, seed=seed, overwrite=False, save_dir=new_save_dir)

        scrna_feature_file = extract_scrna_feature2(scrna_file, node_file, cell_num=cell_num, seed=seed, overwrite=False, just_node=False, save_dir=new_save_dir)

        # scrna_feature_file = extract_scrna_feature2(scrna_file, node_file, cell_num=cell_num, seed=seed, overwrite=False, just_node=False, save_dir=new_save_dir, method='genie3')

        cross_graph_file, cross_node_file, cross_atac_file, aross_scrna_file = \
           build_cross_graph(graph_file, atac_feature_file, scrna_feature_file, overwrite=True, save_dir=new_save_dir)

        gc.collect()
