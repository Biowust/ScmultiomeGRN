library(Seurat)
library(Azimuth)

RunAzimuth_local <- function(
    query,
    reference,
    query.modality = "RNA",
    annotation.levels = NULL,
    umap.name = "ref.umap",
    do.adt = FALSE,
    verbose = TRUE,
    assay = NULL,
    k.weight = 50,
    n.trees = 20,
    mapping.score.k = 100, 
    ...
) {
  CheckDots(...)
  assay <- assay %||% DefaultAssay(query)
  if (query.modality == "ATAC"){
    query <- RunAzimuthATAC(query = query, 
                            reference = reference, 
                            annotation.levels = annotation.levels, 
                            umap.name = umap.name,
                            verbose = verbose, 
                            assay = assay,
                            k.weight = k.weight,
                            n.trees = n.trees, 
                            mapping.score.k = mapping.score.k, 
                            ...)
  } else {
    if (dir.exists(reference)) {
      reference <- LoadReference(reference)$map
    } else {
      reference <- tolower(reference)
      if (reference %in% InstalledData()$Dataset) {
        # only get the `map` object since no plotting is performed
        reference <- LoadData(reference, type = "azimuth")$map
      } else if (reference %in% AvailableData()$Dataset) {
        InstallData(reference)
        # only get the `map` object since no plotting is performed
        reference <- LoadData(reference, type = "azimuth")$map
      } else {
        possible.references <- AvailableData()$Dataset[grepl("*ref", AvailableData()$Dataset)]
        print("Choose one of:")
        print(possible.references)
        stop(paste("Could not find a reference for", reference))
      }
      # handle expected new parameters in uwot models beginning in v0.1.13
      if (!"num_precomputed_nns" %in% names(Misc(reference[["refUMAP"]])$model)) {
        Misc(reference[["refUMAP"]], slot="model")$num_precomputed_nns <- 1
      }
      key.pattern = "^[^_]*_"
      new.colnames <- gsub(pattern = key.pattern, 
                           replacement = Key(reference[["refDR"]]), 
                           x = colnames(Loadings(
                             object = reference[["refDR"]],
                             projected = FALSE)))
      colnames(Loadings(object = reference[["refDR"]], 
                        projected = FALSE)) <- new.colnames
    }
    dims <- as.double(slot(reference, "neighbors")$refdr.annoy.neighbors@alg.info$ndim)
    if (isTRUE(do.adt) && !("ADT" %in% Assays(reference))) {
      warning("Cannot impute an ADT assay because the reference does not have antibody data")
      do.adt = FALSE
    }
    reference.version <- ReferenceVersion(reference)
    azimuth.version <- as.character(packageVersion(pkg = "Azimuth"))
    seurat.version <- as.character(packageVersion(pkg = "Seurat"))
    meta.data <- names(slot(reference, "meta.data"))
    
    # is annotation levels are not specify, gather all levels of annotation
    if (is.null(annotation.levels)) {
      annotation.levels <- names(slot(object = reference, name = "meta.data"))
      annotation.levels <- annotation.levels[!grepl(pattern = "^nCount", x = annotation.levels)]
      annotation.levels <- annotation.levels[!grepl(pattern = "^nFeature", x = annotation.levels)]
      annotation.levels <- annotation.levels[!grepl(pattern = "^ori", x = annotation.levels)]
    }
    
    # Change the file path based on where the query file is located on your system.
    query <- ConvertGeneNames(
      object = query,
      reference.names = rownames(x = reference),
      homolog.table = 'homologs.rds'
    )
    
    # Calculate nCount_RNA and nFeature_RNA if the query does not
    # contain them already
    if (!all(c("nCount_RNA", "nFeature_RNA") %in% c(colnames(x = query[[]])))) {
      calcn <- as.data.frame(x = Seurat:::CalcN(object = query[["RNA"]]))
      colnames(x = calcn) <- paste(
        colnames(x = calcn),
        "RNA",
        sep = '_'
      )
      query <- AddMetaData(
        object = query,
        metadata = calcn
      )
      rm(calcn)
    }
    
    # Calculate percent mitochondrial genes if the query contains genes
    # matching the regular expression "^MT-"
    if (any(grepl(pattern = '^MT-', x = rownames(x = query)))) {
      query <- PercentageFeatureSet(
        object = query,
        pattern = '^MT-',
        col.name = 'percent.mt',
        assay = assay
      )
    }
    # Find anchors between query and reference
    anchors <- FindTransferAnchors(
      reference = reference,
      query = query,
      k.filter = NA,
      reference.neighbors = "refdr.annoy.neighbors",
      reference.assay = "refAssay",
      query.assay = "RNA",
      reference.reduction = "refDR",
      normalization.method = "SCT",
      features = rownames(Loadings(reference[["refDR"]])),
      dims = 1:dims,
      n.trees = n.trees,
      mapping.score.k = mapping.score.k,
      verbose = verbose
    )
    # Transferred labels are in metadata columns named "predicted.*"
    # The maximum prediction score is in a metadata column named "predicted.*.score"
    # The prediction scores for each class are in an assay named "prediction.score.*"
    # The imputed assay is named "impADT" if computed
    refdata <- lapply(X = annotation.levels, function(x) {
      reference[[x, drop = TRUE]]
    })
    names(x = refdata) <- annotation.levels
    
    if (isTRUE(do.adt)) {
      refdata[["impADT"]] <- GetAssayData(
        object = reference[["ADT"]],
        slot = "data"
      )
    }
    
    query <- TransferData(
      reference = reference,
      query = query,
      dims = 1:dims,
      anchorset = anchors,
      refdata = refdata,
      n.trees = 20,
      store.weights = TRUE,
      k.weight = k.weight,
      verbose = verbose
    )
    # Calculate the embeddings of the query data on the reference SPCA
    query <- IntegrateEmbeddings(
      anchorset = anchors,
      reference = reference,
      query = query,
      reductions = "pcaproject",
      reuse.weights.matrix = TRUE,
      verbose = verbose
    )
    # Calculate the query neighbors in the reference
    # with respect to the integrated embeddings
    query[["query_ref.nn"]] <- FindNeighbors(
      object = Embeddings(reference[["refDR"]]),
      query = Embeddings(query[["integrated_dr"]]),
      return.neighbor = TRUE,
      l2.norm = TRUE,
      verbose = verbose
    )
    # The reference used in the app is downsampled compared to the reference on which
    # the UMAP model was computed. This step, using the helper function NNTransform,
    # corrects the Neighbors to account for the downsampling.
    query <- NNTransform(
      object = query,
      meta.data = reference[[]]
    )
    # Project the query to the reference UMAP.
    query[[umap.name]] <- RunUMAP(
      object = query[["query_ref.nn"]],
      reduction.model = reference[["refUMAP"]],
      reduction.key = 'UMAP_',
      verbose = verbose
    )
    # Calculate mapping score and add to metadata
    query <- AddMetaData(
      object = query,
      metadata = MappingScore(anchors = anchors, ndim = dims),
      col.name = "mapping.score"
    )
  }
  return(query)
}

NNTransform <- function(
    object,
    meta.data,
    neighbor.slot = "query_ref.nn",
    key = 'ori.index'
) {
  on.exit(expr = gc(verbose = FALSE))
  ind <- Indices(object[[neighbor.slot]])
  ori.index <- t(x = sapply(
    X = 1:nrow(x = ind),
    FUN = function(i) {
      return(meta.data[ind[i, ], key])
    }
  ))
  rownames(x = ori.index) <- rownames(x = ind)
  slot(object = object[[neighbor.slot]], name = "nn.idx") <- ori.index
  return(object)
}