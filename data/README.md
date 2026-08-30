# Data Directory

This directory should contain:

## Training Data
- `merged_24k_enhanced_mixed.parquet` - Training dataset (24k samples)

## Validation Data  
- `balanced_4cat_50each_natural_mixed.parquet` - Validation dataset (200 samples)

## Data Format
Each sample in parquet file should have:
- `chief_complaint`: Patient's main complaint (主诉)
- `history_present_illness`: Present illness history (现病史)
- `past_medical_history`: Past medical history (既往史)
- `physical_examination`: Physical exam findings (体格检查)
- `auxiliary_examination`: Lab/imaging results (辅助检查)
- `diagnosis`: Ground truth diagnosis (诊断)
- `category`: Disease category (疾病类别)

## Download
Due to medical data privacy, the dataset is not included.
Please prepare your own medical dialogue dataset in the above format.
