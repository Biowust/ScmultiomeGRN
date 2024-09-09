import os
import sys
import h5py
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse as sp_sparse
from scipy import io as sio
from collections import OrderedDict, defaultdict
from sklearn import metrics

PERL_LOC = os.path.join(os.path.dirname(sys.executable), "perl")
PERL_LOC = PERL_LOC if os.path.exists(PERL_LOC) else "perl"

FIMO_LOC = os.path.join(os.path.dirname(sys.executable), "fimo")
FIMO_LOC = FIMO_LOC if os.path.exists(FIMO_LOC) else "fimo"


def exec_command(command, verbose=False):
    if verbose:
        print(command)
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ, shell=True)

    returncode = p.poll()  # 用于检查子进程是否已经结束
    if verbose:
        while returncode is None:
            out = p.stdout.readline().strip().decode()
            if len(out)!=0:
                print(out)
            returncode = p.poll()
        out = p.stdout.readline().strip().decode()
        if len(out) != 0:
            print(out)
    stdout, stderr = p.communicate()
    if p.returncode:
        raise Exception('failed: on %s\n%s\n%s' % (command, stderr, stdout.decode()))
    if stdout is not None:
        return stdout.decode().strip()


def exec_commands(commands, batch_size=None, desc="exec"):
    batch_size = os.cpu_count()//2 if batch_size is None else batch_size
    for i in tqdm(range(0, len(commands), batch_size), desc=desc):
        j = min(len(commands), i+batch_size)
        ps = [(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ, shell=True), command) for command in commands[i:j]]
        for p, cmd in ps:
            stdout, stderr = p.communicate()
            if p.returncode:
                raise Exception('failed: on %s\n%s\n%s' % (cmd, stderr, stdout.decode()))


def metric_fn(predict, label, threshold=None):
    list_fpr, list_tpr, _ = metrics.roc_curve(y_true=label, y_score=predict)
    auroc = metrics.auc(list_fpr, list_tpr)

    # auroc = roc_auc_score(label, predict)

    list_precision, list_recall, _ = metrics.precision_recall_curve(y_true=label, y_score=predict)
    aupr = metrics.auc(list_recall, list_precision)

    ap_score = metrics.average_precision_score(label, predict)
    if threshold is None:
        threshold = float(np.nanmedian(predict))
    predict_binary = predict >= threshold
    temp = predict_binary + label
    acc = (sum(temp == 2) + sum(temp == 0)) / len(temp)
    precision = metrics.precision_score(y_true=label, y_pred=predict_binary)
    recall = metrics.recall_score(y_true=label, y_pred=predict_binary)
    metric = {"aupr": aupr,
              "auroc": auroc,
              "ap": ap_score,
              "acc": acc,
              "precision": precision,
              "recall": recall,
              "threshold": threshold}
    curve = {"fpr": list(list_fpr),
             "tpr": list(list_tpr),
             "precision": list(list_precision),
             "recall": list(list_recall)}
    return metric, curve

def clean_chromosome_name(name):
    return name.replace("\t", "_").replace(":", "_").replace("-", "_").replace("'", "").replace('"', "")

def ATAC_filter_from_CSV(filename, threshold=0.1, sep=",", overwrite=False, save_dir="."):
    tmp_file = os.path.join(save_dir, "ATAC_peak.temp")
    output_file = os.path.join(save_dir, "ATAC_peak.bed")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(filename) as f:
        with open(tmp_file, "w") as writer:
            columns = f.readline().split(sep)
            for line in f:
                tmp = line.strip().split(sep)
                chro, start, end = clean_chromosome_name(tmp[0]).split("_")
                n = sum(map(lambda x:int(x)>0, tmp[1:]))
                total = len(tmp[1:])
                rate = n/total
                if rate>threshold:
                    writer.write(f"{chro}\t{start}\t{end}\t{rate}\n")
    # os.system(f"sort-bed {tmp_file} > {output_file}")
    command = f"sort-bed {tmp_file} > {output_file}"
    exec_command(command, verbose=True)
    exec_command(f"rm {os.path.abspath(tmp_file)}", verbose=True)
    return output_file


def load_mtx(data_dir, use_variable_feature=False, additional_genes=None):
    data = sio.mmread(os.path.join(data_dir, "matrix.mtx"))
    cell_names = pd.read_csv(os.path.join(data_dir, 'barcodes.tsv'), names=['barcode'])['barcode'].values
    try:
        gene_names = pd.read_csv(os.path.join(data_dir, 'genes.tsv'), names=['gene'])['gene'].values
    except:
        gene_names = pd.read_csv(os.path.join(data_dir, 'peaks.tsv'), names=['gene'])['gene'].values
    ans = pd.DataFrame.sparse.from_spmatrix(data)
    ans.index = gene_names
    ans.columns = cell_names
    if use_variable_feature and os.path.exists(os.path.join(data_dir, 'var_features.tsv')):
        print("using variable feature")
        index = set(pd.read_csv(os.path.join(data_dir, 'var_features.tsv'), names=['gene'])['gene'].values)
        if additional_genes is not None:
            index = index.union(additional_genes)
        index = list(index.intersection(ans.index))
        ans = ans.loc[index]
    return ans


def ATAC_filter_from_MTX(filename, threshold=0.1, sep=",", overwrite=False, save_dir="."):
    tmp_file = os.path.join(save_dir, "ATAC_peak.temp")
    output_file = os.path.join(save_dir, "ATAC_peak.bed")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_dir = os.path.dirname(filename)
    data = load_mtx(data_dir)
    with open(tmp_file, "w") as writer:
        for idx, row in tqdm(data.iterrows(), total=len(data), desc='filter peaks'):
            chro, start, end = clean_chromosome_name(idx).split("_")
            n = sum(row>0)
            total = len(row)
            rate = n/total
            if rate>threshold:
                writer.write(f"{chro}\t{start}\t{end}\t{rate}\n")
    # os.system(f"sort-bed {tmp_file} > {output_file}")
    command = f"sort-bed {tmp_file} > {output_file}"
    exec_command(command, verbose=True)
    exec_command(f"rm {os.path.abspath(tmp_file)}", verbose=True)
    return output_file

def extract_chromosome(bed_file, genome_file="../data_resource/hg19/index", overwrite=False, save_dir="."):
    output_file = os.path.join(save_dir, "peak.fasta")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    script_file = os.path.join(os.path.dirname(__file__), "GetSequence.pl")
    assert os.path.exists(bed_file)
    assert os.path.exists(genome_file)
    assert os.path.exists(script_file)
    command = f"{PERL_LOC} {script_file} {os.path.abspath(bed_file)} {os.path.abspath(output_file)} {os.path.abspath(genome_file)}"
    exec_command(command, verbose=True)
    return output_file


def fimo(fasta_file, threshold=1e-4,  motif_dir="../data_resource/human_HOCOMO", overwrite=False, save_dir="."):
    # 匹配motif
    output_dir = os.path.join(save_dir, "fimo-res", str(threshold))
    motif_files = sorted([os.path.join(motif_dir, file) for file in os.listdir(motif_dir) if file.endswith(".meme")])
    run_commands = []
    mv_commands = []
    for motif_file in tqdm(motif_files, desc="fimo"):
        tmp = os.path.basename(motif_file).split(".")
        name = f"{tmp[0].split('_HUMAN')[0]}.{'.'.join(tmp[2:-1])}"
        # tmp = os.path.basename(motif_file).split("_HUMAN.H10MO.")
        # name = ".".join([tmp[0], tmp[1].split(".meme")[0]])
        tmp_dir = os.path.abspath(os.path.join(output_dir, "tmp", name))
        output_file = os.path.abspath(os.path.join(output_dir, f"{name}.txt"))
        if os.path.exists(output_file) and not overwrite:
            continue
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        command = f"{FIMO_LOC} --oc {os.path.abspath(tmp_dir)} --thresh {threshold} --no-qvalue {os.path.abspath(motif_file)} {os.path.abspath(fasta_file)}"
        src_file = os.path.join(tmp_dir, 'fimo.tsv')
        run_commands.append(command)
        mv_commands.append(f"mv {src_file} {output_file}")
    exec_commands(run_commands, desc="fimo run")
    exec_commands(mv_commands, desc="fimo mv")
    exec_command(f"rm {os.path.abspath(os.path.join(output_dir, 'tmp'))} -rf", verbose=True)
    return output_dir


def get_TFBS_region(fimo_dir, threshold=1e-6, overwrite=False, save_dir="."):
    txt_files = sorted([file for file in os.listdir(fimo_dir) if file.endswith(".txt")])
    output_dir = os.path.join(save_dir, "TFBS_region")
    tmp_dir = os.path.join(output_dir, "tmp")
    output_file = os.path.join(output_dir, "atac_tfbs_region_all.txt")
    if os.path.exists(output_file) and not overwrite:
        return output_file

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    detail_dir = os.path.join(output_dir, "detail")
    if not os.path.exists(detail_dir):
        os.makedirs(detail_dir)
    for file in tqdm(txt_files, desc="get TFBS region"):
        outputs = []
        name = file[:-4]
        detail_file = os.path.join(detail_dir, f"{name}.bed")
        if os.path.exists(detail_file) and not overwrite:
            continue
        with open(os.path.join(fimo_dir, file)) as f:
            head = f.readline().strip()
            if len(head)==0:
                continue
            assert head.startswith("motif_id")
            for line in f:
                line = line.strip()
                if len(line)==0:
                    break
                tmp = line.split('\t')
                if float(tmp[7])>threshold:
                    continue
                tf = tmp[0].split("_")[0]
                chro, start, end = tmp[2].split("-")[:3]
                abs_start = int(start)+int(tmp[3])-1
                abs_end = int(start)+int(tmp[4])-1
                outputs.append(f"{chro}\t{abs_start}\t{abs_end}\t{tf}\n")
        if len(outputs)!=0:
            tmp_file = os.path.join(tmp_dir, f"{name}.tmp")
            with open(tmp_file, "w") as writer:
                writer.writelines(outputs)
            exec_command(f"sort-bed {os.path.abspath(tmp_file)} > {os.path.abspath(detail_file)}")
            exec_command(f"rm {os.path.abspath(tmp_file)}")
    tmp_file = os.path.join(tmp_dir, "tfbs.tmp")
    with open(tmp_file, "w") as writer:
        for file in os.listdir(detail_dir):
            if file.endswith(".bed"):
                with open(os.path.join(detail_dir, file)) as f:
                    for line in f:
                        writer.write(line)
    exec_command(f"sort-bed {os.path.abspath(tmp_file)} > {os.path.abspath(output_file)}", verbose=True)
    exec_command(f"rm {os.path.abspath(tmp_dir)} -rf", verbose=True)
    return output_file


def get_TFBS_from_ATAC(atac_file, motif_dir, genome_file, sep=',',
                       peak_threshold=0.1, fimo_threshold=1e-4, tfbs_threshold=1e-6,
                       overwrite=False, save_dir="."):

    output_dir = os.path.join(save_dir, "region")
    output_file = os.path.join(output_dir, "atac_tfbs_region_all_tmp.txt")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_dir = os.path.join(save_dir, "tmp")
    if atac_file.endswith(".csv"):
        bed_file = ATAC_filter_from_CSV(atac_file, sep=sep, threshold=peak_threshold, overwrite=overwrite, save_dir=save_dir)
    elif atac_file.endswith(".mtx"):
        bed_file = ATAC_filter_from_MTX(atac_file, sep=sep, threshold=peak_threshold, overwrite=overwrite, save_dir=save_dir)
    else:
        raise NotImplementedError
    fasta_file = extract_chromosome(bed_file, genome_file, overwrite=overwrite, save_dir=save_dir)
    fimo_dir = fimo(fasta_file, threshold=fimo_threshold, motif_dir=motif_dir, overwrite=overwrite, save_dir=save_dir)
    region_file = get_TFBS_region(fimo_dir, threshold=tfbs_threshold, overwrite=overwrite, save_dir=save_dir)
    shutil.copy(region_file, output_file)
    return output_file


def get_TFBS_from_promoter(promoter_file, motif_dir="../data_resource/human_HOCOMO", save_dir=".", overwrite=False):
    output_dir = os.path.join(save_dir, "region")
    output_file = os.path.join(output_dir, "gencode.TF.promoter.region.txt")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    motif_name = [file.split("_")[0] for file in os.listdir(motif_dir) if file.endswith(".meme")]
    print(f"total TF motif: {len(motif_name)}")
    motif_name = set(motif_name)
    tmp_file = os.path.join(output_dir, "gencode.TF.promoter.tmp")
    cnt = 0
    with open(tmp_file, "w") as writer:
        with open(promoter_file) as f:
            columns = f.readline()
            assert columns.startswith("chr	")
            for line in f:
                tmp = line.strip().split("\t")
                gene_name = tmp[4]
                if gene_name in motif_name:
                    writer.write(line)
                    cnt += 1
    print(f"TF motif in promoter: {cnt}")
    exec_command(f"sort-bed {os.path.abspath(tmp_file)} > {os.path.abspath(output_file)}", verbose=True)
    exec_command(f"rm {os.path.abspath(tmp_file)}", verbose=True)
    return output_file


def get_overlap_region(atac_region_file, promoter_region_file, overwrite=False, save_dir="."):
    output_dir = os.path.join(save_dir, "region")
    output_file = os.path.join(output_dir, "atac.overlay.region.txt")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = f"bedops -e -20% {os.path.abspath(atac_region_file)} {os.path.abspath(promoter_region_file)} > {os.path.abspath(output_file)}"
    exec_command(command, verbose=True)

    cnt = 0
    with open(output_file) as f:
        for line in f:
            cnt += 1
    print(f"{cnt} TFBS overlay with TF-Promoter region")
    return output_file


def get_contained_region(atac_region_file, promoter_region_file, overwrite=False, save_dir="."):
    output_dir = os.path.join(save_dir, "region")
    bed_file = os.path.join(output_dir, "atac.contained.region.txt")
    if os.path.exists(bed_file) and not overwrite:
        return bed_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = f"bedops -e -20% {os.path.abspath(atac_region_file)} {os.path.abspath(promoter_region_file)} > {os.path.abspath(bed_file)}"
    exec_command(command, verbose=True)

    cnt = 0
    with open(bed_file) as f:
        for line in f:
            cnt += 1
    print(f"{cnt} TFBS in TF-Promoter region")
    return bed_file


def build_Graph(atac_region_file, promoter_region_file, overwrite=False, save_dir="."):
    output_dir = os.path.join(save_dir, "origin_graph")
    output_file = os.path.join(output_dir, "adj.txt")
    node_file = os.path.join(output_dir, "node.txt")
    if os.path.exists(output_file) and os.path.exists(node_file) and not overwrite:
        return output_file, node_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bed_file = get_contained_region(atac_region_file, promoter_region_file, overwrite=overwrite, save_dir=save_dir)
    querys = defaultdict(list)

    with open(bed_file) as f:
        for line in f:
            items = line.strip().split('\t')
            querys[items[0]].append((int(items[1]), int(items[2]), items[3]))

    hash_set = set()
    tf_set = OrderedDict()
    with open(promoter_region_file) as f:
        for line in f:
            tmp = line.strip().split('\t')
            chro, start, end, strand, tf = tmp[:5]
            start = int(start)
            end = int(end)
            cnt = 0
            for query in querys.get(chro, []):
                if (query[0]>=start and query[1]<=end) :#or (query[0]<=start and end<=query[1]):
                    cnt += 1
                    hash_set.add(f"{tf}\t{query[2]}")
                    hash_set.add(f"{query[2]}\t{tf}")
            if cnt>0:
                tf_set[tf] = len(tf_set)+1
    print(f"Contained TF node num: {len(tf_set)}")

    tf_set = sorted(tf_set)
    total = 0
    ans = ["\t".join(["TF"] + [row for row in tf_set])]
    for row in tf_set:
        tmp = [row]
        for col in tf_set:
            key = f"{row}\t{col}"
            if key in hash_set:
                tmp.append("1")
                total += 1
            else:
                tmp.append("0")
        ans.append("\t".join(tmp))
    print(f"edge num: {total}")
    print(f"density: {total/len(tf_set)/len(tf_set)}")

    with open(output_file, "w") as f:
        for line in ans:
            f.write(f"{line}\n")

    with open(node_file, "w") as f:
        for line in tf_set:
            f.write(f"{line}\n")

    return output_file, node_file


def write_10X_h5(filename, matrix, features, barcodes, genome='GRCh38', datatype='Peak'):
    """Write 10X HDF5 files, support both gene expression and peaks."""
    f = h5py.File(filename, 'w')
    if datatype == 'Peak':
        M = sp_sparse.csc_matrix(matrix, dtype=np.int8)
    else:
        M = sp_sparse.csc_matrix(matrix, dtype=np.float32)
    B = np.array(barcodes, dtype='|S200')
    P = np.array(features, dtype='|S100')
    GM = np.array([genome] * len(features), dtype='|S10')
    FT = np.array([datatype] * len(features), dtype='|S100')
    AT = np.array(['genome'], dtype='|S10')
    mat = f.create_group('matrix')
    mat.create_dataset('barcodes', data=B)
    mat.create_dataset('data', data=M.data)
    mat.create_dataset('indices', data=M.indices)
    mat.create_dataset('indptr', data=M.indptr)
    mat.create_dataset('shape', data=M.shape)
    fet = mat.create_group('features')
    fet.create_dataset('_all_tag_keys', data=AT)
    fet.create_dataset('feature_type', data=FT)
    fet.create_dataset('genome', data=GM)
    fet.create_dataset('id', data=P)
    fet.create_dataset('name', data=P)
    f.close()


def convert_atac_csv_to_10X_h5(atac_file, cell_num=None, seed=666, lift=False, lift_chain=None, genome='GRCh38', datatype='Peak', save_dir=".", overwrite=False):
    output_dir = os.path.join(save_dir, "atac_data")
    output_file = os.path.join(output_dir, f"{datatype}.h5")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if atac_file.endswith(".csv"):
        data = pd.read_csv(atac_file, sep=',', index_col=0)
        if cell_num is not None:
            data = data.sample(n=cell_num, random_state=seed, axis=1, replace=False)
        csc_data = sp_sparse.csc_matrix(data.values)
    elif atac_file.endswith(".mtx"):
        data = load_mtx(os.path.dirname(atac_file))
        if cell_num is not None:
            data = data.sample(n=cell_num, random_state=seed, axis=1, replace=False)
        csc_data = data.sparse.to_coo().tocsc()
    else:
        raise NotImplementedError
    # coo_data = sp_sparse.coo_matrix(data.values)
    # coordinate_with_count = np.array([coo_data.row + 1, coo_data.col + 1, coo_data.data])
    # Num = len(coo_data.data)
    # P_num = np.size(data, 0)
    # C_num = np.size(data, 1)
    features = [clean_chromosome_name(name) for name in data.axes[0]]
    if lift:
        features = lift_hg19to38(features, lift_chain=lift_chain)
    barcodes = data.axes[1]

    # features: chr1_10279_10779
    # barcodes: singles-BM1214-MCP-frozen-160128-1
    write_10X_h5(filename=output_file, matrix=csc_data, features=features,
                 barcodes=barcodes, genome=genome, datatype=datatype)
    return output_file

def extract_data_from_gene(file, gene_name_file, output_file):
    gene = pd.read_csv(gene_name_file, names=['node'])['node'].values
    if file.endswith(".csv"):
        data = pd.read_csv(file, index_col=0, sep=',')
    else:
        data = load_mtx(os.path.dirname(file))
    node = data.index.intersection(gene)
    new_data = data.loc[node]
    new_data.to_csv(output_file, sep=',')
    return output_file

def extract_atac_feature(atac_file, node_file, just_node=False, lift=False, lift_chain=None, cell_num=None, seed=666, genedistance=1000, genome='GRCh38', datatype='Peak', save_dir=".", overwrite=False, remove_cache=False):
    output_dir = os.path.join(save_dir, "origin_graph")
    output_file = os.path.join(output_dir, "atac_rp_score_feature.txt")

    tmp_dir = os.path.abspath(os.path.join(save_dir, "tmp"))
    output_tmp_file = os.path.join(tmp_dir, "TF_rp_score.txt")
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # if just_node:
    #     if not os.path.exists(tmp_dir):
    #         os.makedirs(tmp_dir)
    #     atac_tmp_file = os.path.join(tmp_dir, "just_atac.csv")
    #     atac_file = extract_data_from_gene(atac_file, node_file, atac_tmp_file)

    h5_file = convert_atac_csv_to_10X_h5(atac_file, cell_num=cell_num, seed=seed, lift=lift, lift_chain=lift_chain, genome=genome, datatype=datatype, save_dir=tmp_dir, overwrite=overwrite)
    script_file = os.path.join(os.path.dirname(__file__), "MAESTRO.sh")
    r_script_file = os.path.join(os.path.dirname(__file__), "for_TF_gene_rp_score.R")

    assert os.path.exists(script_file)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    shutil.copy(node_file, os.path.join(tmp_dir, os.path.basename(node_file)))
    shutil.copy(h5_file, os.path.join(tmp_dir, os.path.basename(h5_file)))
    shutil.copy(script_file, os.path.join(tmp_dir, os.path.basename(script_file)))
    shutil.copy(r_script_file, os.path.join(tmp_dir, os.path.basename(r_script_file)))

    docker_h5_file = os.path.basename(h5_file)
    docker_node_file = os.path.basename(node_file)
    output_prefix = f"d{genedistance}"
    species = genome
    owner = f"{exec_command('id -u')}:{exec_command('id -g')}"
    overwrite_flag = "1" if overwrite else "0"

    command = rf"docker run -v {tmp_dir}:/mnt winterdongqing/maestro:1.2.1 bash -e /mnt/MAESTRO.sh {docker_h5_file} {output_prefix} {genedistance} {species} {owner} {docker_node_file} {overwrite_flag}"
    exec_command(command, verbose=True)
    data = pd.read_csv(output_tmp_file, sep='\t')
    node = pd.read_csv(node_file, names=["gene"])
    data.index = node['gene'].values
    data.to_csv(output_file, sep='\t')
    if remove_cache:
        shutil.rmtree(tmp_dir)
    return output_file


def extract_scrna_feature2(scrna_file, node_file, method="grnboost2", just_node=False, cell_num=None, seed=666, save_dir=".", overwrite=False, use_variable_feature=True):
    from arboreto.algo import genie3, grnboost2
    output_dir = os.path.join(save_dir, "origin_graph")
    if just_node:
        output_file = os.path.join(output_dir, f"scRNA_{method}_just_node_feature.txt")
    else:
        output_file = os.path.join(output_dir, f"scRNA_{method}_feature.txt")
    tmp_dir = os.path.abspath(os.path.join(save_dir, "tmp"))
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if method=="grnboost2":
        func = grnboost2
    elif method=="genie3":
        func = genie3
    else:
        raise NotImplementedError

    genes = pd.read_csv(node_file, names=['gene'])['gene'].values
    scrna_tmp_file = os.path.join(tmp_dir, "just_scrna.csv") if just_node else os.path.join(tmp_dir, "scrna.csv")
    if not os.path.exists(scrna_tmp_file) or overwrite:
        if scrna_file.endswith(".csv"):
            data = pd.read_csv(scrna_file, index_col=0, sep=',')
        else:
            data = load_mtx(os.path.dirname(scrna_file),
                            use_variable_feature=use_variable_feature,
                            additional_genes=genes)
        if cell_num is not None:
            data = data.sample(n=cell_num, random_state=seed, axis=1, replace=False)
        genes = sorted(set(genes).intersection(data.index))
        if just_node:
            data = data.loc[genes]
        data.to_csv(scrna_tmp_file)
    # exit()
    data = pd.read_csv(scrna_tmp_file, index_col=0)
    genes = sorted(set(genes).intersection(data.index))
    other_gene = list(set(data.index)-set(genes))
    all_gene = list(genes)+other_gene
    genes2id = {key:i for i, key in enumerate(all_gene)}
    association = func(expression_data=data.T, tf_names=list(genes), seed=0, verbose=True)
    network = pd.DataFrame(np.zeros((len(genes), len(genes2id))), index=genes, columns=all_gene)
    for idx, row in tqdm(association.iterrows(), total=len(association)):
        network.loc[row['TF'], row['target']] = row['importance']
    network.to_csv(output_file, sep='\t')
    return output_file


def extract_scrna_feature(scrna_file, node_file=None, just_node=True, save_dir=".", overwrite=False):
    output_dir = os.path.join(save_dir, "origin_graph")
    output_file = os.path.join(output_dir, "scRNA_genie3_feature.txt")
    tmp_dir = os.path.abspath(os.path.join(save_dir, "tmp"))
    if os.path.exists(output_file) and not overwrite:
        return output_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if just_node:
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        scrna_tmp_file = os.path.join(tmp_dir, "just_scrna.csv")
        scrna_file = extract_data_from_gene(scrna_file, node_file, scrna_tmp_file)

    script_file = os.path.join(os.path.dirname(__file__), "for_TF_genie3_score.py")
    python_loc = sys.executable
    command = f"{python_loc} {script_file} --input_file {os.path.abspath(scrna_file)} --output_file {os.path.abspath(output_file)}"
    exec_command(command, verbose=True)

    # from for_TF_genie3_score import GENIE3
    # data = pd.read_csv(scrna_file, index_col=0)
    # gene_names = data.axes[0].tolist()
    # expr_data = data.T.values
    # res = GENIE3(expr_data, gene_names, nthreads=1)
    # np.savetxt(output_file, res)

    return output_file


def build_cross_graph(graph_file, atac_feature_file, scrna_feature_file, save_dir=".", overwrite=False):
    output_dir = save_dir

    output_graph_file = os.path.join(output_dir, "x_graph.txt")
    output_node_file = os.path.join(output_dir, "x_node.txt")
    output_atac_feature_file = os.path.join(output_dir, f"x_{os.path.basename(atac_feature_file)}")
    output_scrna_feature_file = os.path.join(output_dir, f"x_{os.path.basename(scrna_feature_file)}")

    if os.path.exists(output_graph_file) and os.path.exists(output_node_file) and \
        os.path.exists(output_atac_feature_file) and os.path.exists(output_scrna_feature_file) and \
        not overwrite:
        return output_graph_file, output_node_file, output_atac_feature_file, output_scrna_feature_file

    atac_feature = pd.read_csv(atac_feature_file, sep='\t', index_col=0)
    scrna_feature = pd.read_csv(scrna_feature_file, sep='\t', index_col=0)
    intersection_index = atac_feature.index.intersection(scrna_feature.index)


    with open(output_node_file, "w") as writer:
        for line in intersection_index:
            writer.write(f"{line}\n")

    adj = pd.read_csv(graph_file, index_col=0, sep='\t')
    x_adj = adj.loc[intersection_index, intersection_index]
    x_adj.to_csv(output_graph_file, sep='\t')

    edge_num = x_adj.values.sum()
    print(f"final graph with {len(intersection_index)} nodes, {edge_num} edges, density: {edge_num/(x_adj.shape[0]*x_adj.shape[1])}")

    x_atac_feature = atac_feature.loc[intersection_index, :]
    x_atac_feature.to_csv(output_atac_feature_file, sep='\t')

    x_scrna_feature = scrna_feature.loc[intersection_index, :]
    x_scrna_feature.to_csv(output_scrna_feature_file, sep='\t')

    return output_graph_file, output_node_file, output_atac_feature_file, output_scrna_feature_file



def run(atac_file, scrna_file, save_dir):

    motif_dir = "/home/jm/PycharmProjects/xjl/DeepTFni/data_resource/human_HOCOMO"
    genome_file = "/home/jm/PycharmProjects/xjl/DeepTFni/data_resource/download_genome/hg19/index2"
    promoter_file = "/home/jm/PycharmProjects/xjl/DeepTFni/data_resource/gencode.v19.ProteinCoding_gene_promoter.txt"

    atac_region_file = get_TFBS_from_ATAC(atac_file, motif_dir, genome_file, overwrite=False, save_dir=save_dir)

    promoter_region_file = get_TFBS_from_promoter(promoter_file, overwrite=False, save_dir=save_dir)
    graph_file, node_file = build_Graph(atac_region_file, promoter_region_file, overwrite=True, save_dir=save_dir)

    atac_feature_file = extract_atac_feature(atac_file, node_file, overwrite=True, save_dir=save_dir)

    scrna_feature_file = extract_scrna_feature(scrna_file, node_file, just_node=True, overwrite=True, save_dir=save_dir)
    cross_graph_file, cross_node_file, cross_atac_file, aross_scrna_file = \
        build_cross_graph(graph_file, atac_feature_file, scrna_feature_file, overwrite=True, save_dir=save_dir)

def lift_hg19to38(chromosome_names, lift_chain=None):

    ans = chromosome_names
    return ans


if __name__=="__main__":
    atac_file = "/home/jm/PycharmProjects/xjl/DeepTFni/ATAC_input/Megakaryocytes.csv"
    scrna_file = "/home/jm/PycharmProjects/xjl/DeepTFni/scRNA_input/Megakaryocytes.csv"
    save_dir = "output/Megakaryocytes"
    save_dir = 'output/CD'

    run(atac_file, scrna_file, save_dir)



