# Copy outside the repository; substitute authorized external data paths.
export CRC_QPCR_ROOT=/external/project/qpcr_run
export CRC_QPCR_CONFIG=/external/private/qpcr_config.json
export CRC_TCGA_ROOT=/external/project/tcga_run
export CRC_TCGA_DB_ROOT=/external/databases/TCGA
export CRC_SIGNATURE_GMT=/external/frozen/signatures/frozen_signatures.gmt
export CRC_SIGNATURE_ROOT=/external/project/signatures_run
export CRC_SIGNATURE_SOURCE=/external/upstream/signature
export CRC_SIGNATURE_CONFIG=/external/private/signature_config.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
