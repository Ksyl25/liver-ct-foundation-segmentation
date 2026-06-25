# Architecture

Phase 1 implements a local NIfTI viewer and Dice self-check for liver CT segmentation masks.

```mermaid
flowchart LR
    A["NIfTI image + mask"] --> B["loader"]
    B --> C["validation"]
    C --> D["slice viewer"]
    D --> E["overlay"]
    C --> F["Dice self-check"]
```

Future phases will add CT windowing, baseline segmentation, bounding boxes, MedSAM Lite inference, DICOM ingestion and CLI workflows.
