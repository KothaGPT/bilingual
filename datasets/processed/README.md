# Processed Datasets Workspace

This directory is a generic workspace for dataset processing pipelines.
It is organized into stages from raw inputs to final cleaned data.

## Layout

- `raw/`
  - Initial dumps or collected data before any cleaning.
- `cleaned/`
  - Text after basic cleaning (normalization, markup removal, etc.).
- `filtered/`
  - Subset of cleaned data after quality filters (length, noise removal, etc.).
- `final/`
  - Final data ready for training / evaluation.
- `workflow_stats.json`
  - Optional JSON file with statistics about the processing workflow (counts, ratios, etc.).

## Usage

- You can reuse this layout for experiments that are not tied to a single dataset.
- Scripts can read and write from these stages instead of creating ad-hoc folders.
- Keep a short note (e.g. in your experiment or dataset README) describing how you used each stage.
