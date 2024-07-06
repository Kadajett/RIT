# Image Processor

Image Processor is a Rust-based command-line tool that recursively processes images in a directory. It performs various transformations such as resizing, rotating, and flipping, while generating a modified training data JSON file.

## Features

- Recursive image processing
- Image transformations: resize, rotate, flip
- Parallel processing for improved performance
- Modified training data JSON file generation
- Class index preservation based on folder names

## Requirements

- Rust (latest stable version)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/image-processor.git
   cd image-processor
   ```

2. Build the project:
   ```sh
   cargo build --release
   ```

## Usage

1. Prepare your input directory with images organized in subfolders. Each subfolder represents a class.

   Example directory structure:
   ```
   train/
   ├── training_data.json
   ├── ARCIGERA FLOWER MOTH/
   │   ├── 021.jpg
   │   ├── 022.jpg
   │   └── ...
   ├── BLAIRS MOCHA/
   │   ├── 001.jpg
   │   ├── 002.jpg
   │   └── ...
   └── ...
   ```

2. Run the tool:
   ```sh
   cargo run --release
   ```

3. Follow the interactive prompts to specify:
   - Input directory containing images
   - Output directory for modified images
   - Resize dimensions (optional)
   - Rotate angle (optional)
   - Flip options (None, Horizontal, Vertical, Both)

The tool will process the images and generate a new training data JSON file in the output directory.

## Sample Data

Sample `training_data.json` format:

```json
[
    {
        "class index": 0,
        "filepaths": "train/ARCIGERA FLOWER MOTH/021.jpg",
        "labels": "ARCIGERA FLOWER MOTH",
        "data set": "train"
    },
    {
        "class index": 0,
        "filepaths": "train/ARCIGERA FLOWER MOTH/022.jpg",
        "labels": "ARCIGERA FLOWER MOTH",
        "data set": "train"
    },
    {
        "class index": 1,
        "filepaths": "train/BLAIRS MOCHA/001.jpg",
        "labels": "BLAIRS MOCHA",
        "data set": "train"
    }
]
```

## Example Run

```sh
$ cargo run --release
✔ Enter the input directory containing images · ./train
✔ Enter the output directory for modified images · ./train_small
✔ Enter resize dimensions in the format WIDTHxHEIGHT (e.g., 800x600) or press Enter to skip · 64x64
✔ Enter rotate angle (90, 180, 270) or press Enter to skip · 90
✔ Select flip option · Horizontal
Processed 10 images in 2.34s
Average time per image: 0.234s
```

## Performance

The tool logs performance metrics including total processing time, number of processed images, and average time per image.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
