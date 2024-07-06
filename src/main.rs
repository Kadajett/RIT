use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use image::{imageops, DynamicImage, GenericImageView, ImageFormat};
use log::{error, info, warn, LevelFilter};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

const ALLOWED_EXTENSIONS: [&str; 3] = ["png", "jpg", "jpeg"];
const DEFAULT_JPEG_QUALITY: u8 = 85;

#[derive(Parser, Debug)]
#[command(name = "Rit - Rust Image Transformer")]
#[command(about = "Recursively processes images in a directory", long_about = None)]
struct Cli {
    #[arg(short, long, help = "Path to the input directory containing images")]
    input_dir: Option<String>,

    #[arg(short, long, help = "Path to the output directory for modified images")]
    output_dir: Option<String>,

    #[arg(
        short,
        long,
        help = "Path to the existing training data JSON file (optional)"
    )]
    training_json: Option<String>,

    #[arg(long, help = "Preserve original filenames")]
    preserve_filenames: bool,

    #[arg(long, help = "Preserve original file formats")]
    preserve_formats: bool,

    #[arg(long, help = "JPEG quality (1-100)", default_value_t = DEFAULT_JPEG_QUALITY)]
    jpeg_quality: u8,
}

#[derive(Debug)]
struct Config {
    resize: Option<ResizeOption>,
    rotate: Option<f32>,
    flip_horizontal: bool,
    flip_vertical: bool,
}

#[derive(Debug)]
enum ResizeOption {
    Exact(u32, u32),
    Percentage(f32),
    Width(u32),
    Height(u32),
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingData {
    #[serde(rename = "class index")]
    class_index: usize,
    filepaths: String,
    labels: String,
    #[serde(rename = "data set")]
    dataset: String,
}

fn parse_resize(resize: &str) -> Option<ResizeOption> {
    if let Some(percent) = resize.strip_suffix('%') {
        percent.parse().ok().map(ResizeOption::Percentage)
    } else if let Some(width) = resize.strip_suffix('w') {
        width.parse().ok().map(ResizeOption::Width)
    } else if let Some(height) = resize.strip_suffix('h') {
        height.parse().ok().map(ResizeOption::Height)
    } else {
        let parts: Vec<&str> = resize.split('x').collect();
        if parts.len() == 2 {
            parts[0]
                .parse()
                .and_then(|w| parts[1].parse().map(|h| (w, h)))
                .ok()
                .map(|(w, h)| ResizeOption::Exact(w, h))
        } else {
            None
        }
    }
}

fn resize_image(image: &DynamicImage, option: &ResizeOption) -> DynamicImage {
    match option {
        ResizeOption::Exact(width, height) => {
            image.resize_exact(*width, *height, imageops::Lanczos3)
        }
        ResizeOption::Percentage(percent) => {
            let (width, height) = image.dimensions();
            let new_width = (width as f32 * percent / 100.0) as u32;
            let new_height = (height as f32 * percent / 100.0) as u32;
            image.resize(new_width, new_height, imageops::Lanczos3)
        }
        ResizeOption::Width(w) => image.resize(*w, image.height(), imageops::Lanczos3),
        ResizeOption::Height(h) => image.resize(image.width(), *h, imageops::Lanczos3),
    }
}

fn process_image(image: DynamicImage, config: &Config) -> DynamicImage {
    let mut image = image;

    if let Some(resize_option) = &config.resize {
        info!("Resizing image");
        image = resize_image(&image, resize_option);
    }

    if let Some(angle) = config.rotate {
        info!("Rotating image by {} degrees", angle);
        match angle {
            90.0 => image = image.rotate90(),
            180.0 => image = image.rotate180(),
            270.0 => image = image.rotate270(),
            _ => eprintln!("Invalid rotation angle: {}", angle),
        }
    }

    if config.flip_horizontal {
        info!("Flipping image horizontally");
        image = image.fliph();
    }

    if config.flip_vertical {
        info!("Flipping image vertically");
        image = image.flipv();
    }

    image
}

fn save_image(
    image: &DynamicImage,
    output_path: &Path,
    jpeg_quality: u8,
) -> Result<(), image::ImageError> {
    let format = ImageFormat::from_path(output_path)?;
    let mut output_file = File::create(output_path)?;
    image.save_with_format(output_path, format)?;
    Ok(())
}

fn read_existing_training_data(training_json_path: Option<&Path>) -> HashMap<String, usize> {
    let mut label_to_class_index = HashMap::new();

    if let Some(path) = training_json_path {
        if path.exists() {
            let file = File::open(path).expect("Failed to open existing training data file");
            let existing_data: Vec<TrainingData> =
                serde_json::from_reader(file).expect("Failed to read existing training data");
            for data in existing_data {
                label_to_class_index
                    .entry(data.labels.clone())
                    .or_insert(data.class_index);
            }
        }
    }

    label_to_class_index
}

fn process_directory(
    input_dir: &Path,
    output_dir: &Path,
    config: &Config,
    label_to_class_index: &Arc<Mutex<HashMap<String, usize>>>,
    cli: &Cli,
) -> Vec<TrainingData> {
    info!("Processing directory: {:?}", input_dir);
    let training_data: Arc<Mutex<Vec<TrainingData>>> = Arc::new(Mutex::new(Vec::new()));
    let processed_count = Arc::new(Mutex::new(0));

    WalkDir::new(input_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .par_bridge()
        .for_each(|entry| {
            let path = entry.path();
            info!("Found file: {:?}", path);
            let extension = path.extension().and_then(|ext| ext.to_str());
            info!("File extension: {:?}", extension);
            if let Some(ext) = extension {
                if ALLOWED_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    info!("Processing image file: {:?}", path);
                    if let Ok(image) = image::open(path) {
                        let processed_image = process_image(image, config);

                        let relative_path = path.strip_prefix(input_dir).expect("Invalid path");
                        let mut output_path = output_dir.join(relative_path);
                        if !cli.preserve_formats {
                            output_path.set_extension("png");
                        }
                        if !cli.preserve_filenames {
                            if let Some(parent) = output_path.parent() {
                                let new_filename =
                                    format!("{}.png", processed_count.lock().unwrap());
                                output_path = parent.join(new_filename);
                            }
                        }
                        if let Some(parent) = output_path.parent() {
                            fs::create_dir_all(parent).expect("Failed to create output directory");
                        }
                        if let Err(e) = save_image(&processed_image, &output_path, cli.jpeg_quality)
                        {
                            error!("Failed to save image {:?}: {}", output_path, e);
                        } else {
                            info!("Saved image to {:?}", output_path);
                        }

                        let label = path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        let dataset = output_dir
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();

                        let class_index = {
                            let mut label_map = label_to_class_index.lock().unwrap();
                            if !label_map.contains_key(&label) {
                                let new_index = label_map.len();
                                label_map.insert(label.clone(), new_index);
                            }
                            *label_map.get(&label).unwrap()
                        };

                        let mut data = training_data.lock().unwrap();
                        data.push(TrainingData {
                            class_index,
                            filepaths: output_path.display().to_string(),
                            labels: label,
                            dataset,
                        });

                        let mut count = processed_count.lock().unwrap();
                        *count += 1;
                    } else {
                        error!("Failed to open image {:?}", path);
                    }
                }
            }
        });

    let processed_count = Arc::try_unwrap(processed_count)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unlock Mutex");

    println!("Processed {} images", processed_count);

    Arc::try_unwrap(training_data)
        .expect("Failed to unwrap Arc")
        .into_inner()
        .expect("Failed to unlock Mutex")
}

fn prompt_for_config() -> Config {
    let resize: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter resize option (e.g., 800x600, 50%, 800w, 600h) or press Enter to skip")
        .allow_empty(true)
        .interact_text()
        .unwrap();
    let resize = if resize.is_empty() {
        None
    } else {
        parse_resize(&resize)
    };

    let rotate: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter rotation angle in degrees or press Enter to skip")
        .allow_empty(true)
        .validate_with(|input: &String| -> Result<(), &str> {
            if input.is_empty() {
                return Ok(());
            }
            input
                .parse::<f32>()
                .map(|_| ())
                .map_err(|_| "Please enter a valid number")
        })
        .interact_text()
        .unwrap();
    let rotate = if rotate.is_empty() {
        None
    } else {
        Some(rotate.parse().unwrap())
    };

    let flip_options = &["None", "Horizontal", "Vertical", "Both"];
    let flip_selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select flip option")
        .default(0)
        .items(flip_options)
        .interact()
        .unwrap();

    let flip_horizontal = flip_selection == 1 || flip_selection == 3;
    let flip_vertical = flip_selection == 2 || flip_selection == 3;

    Config {
        resize,
        rotate,
        flip_horizontal,
        flip_vertical,
    }
}

fn prompt_for_paths() -> (PathBuf, PathBuf) {
    let input_dir: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter the input directory containing images")
        .interact_text()
        .unwrap();
    let input_dir = Path::new(&input_dir).to_path_buf();

    let output_dir: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter the output directory for modified images")
        .interact_text()
        .unwrap();
    let output_dir = Path::new(&output_dir).to_path_buf();

    (input_dir, output_dir)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::new()
        .filter(None, LevelFilter::Info)
        .init();
    let cli = Cli::parse();

    let (input_dir, output_dir) = match (cli.input_dir.as_ref(), cli.output_dir.as_ref()) {
        (Some(input), Some(output)) => (PathBuf::from(input), PathBuf::from(output)),
        _ => prompt_for_paths(),
    };

    if !input_dir.is_dir() {
        return Err("Input directory does not exist or is not a directory".into());
    }

    let config = prompt_for_config();

    let start_time = Instant::now();

    let training_json_path = cli.training_json.as_deref().map(Path::new);
    let label_to_class_index =
        Arc::new(Mutex::new(read_existing_training_data(training_json_path)));
    let training_data = process_directory(
        &input_dir,
        &output_dir,
        &config,
        &label_to_class_index,
        &cli,
    );

    let duration = start_time.elapsed();

    let json_file_path = output_dir.join("training_data.json");
    let json_file = File::create(&json_file_path)?;
    serde_json::to_writer_pretty(json_file, &training_data)?;

    let processed_count = training_data.len();
    let avg_time_per_image = duration.as_secs_f64() / processed_count as f64;

    if processed_count == 0 {
        warn!("No images processed. Please check your input directory and configuration.");
    }

    info!("Training data JSON file created at {:?}", json_file_path);
    info!("Processed {} images in {:?}", processed_count, duration);
    info!("Average time per image: {:.6} seconds", avg_time_per_image);
    println!("Processed {} images in {:?}", processed_count, duration);
    println!("Average time per image: {:.6} seconds", avg_time_per_image);

    Ok(())
}
