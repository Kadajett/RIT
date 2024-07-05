use clap::Parser;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use image::{imageops, DynamicImage, GenericImageView, ImageFormat};
use log::{error, info, LevelFilter};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "Image Processor")]
#[command(about = "Recursively processes images in a directory", long_about = None)]
struct Cli {}

struct Config {
    resize: Option<(u32, u32)>,
    rotate: Option<u32>,
    flip_horizontal: bool,
    flip_vertical: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingData {
    #[serde(rename = "class index")]
    class_index: usize,
    filepaths: String,
    labels: String,
    #[serde(rename = "data set\r")]
    dataset: String,
}

fn parse_resize(resize: String) -> Option<(u32, u32)> {
    let parts: Vec<&str> = resize.split('x').collect();
    if parts.len() == 2 {
        if let (Ok(width), Ok(height)) = (parts[0].parse(), parts[1].parse()) {
            return Some((width, height));
        }
    }
    None
}

fn process_image(image: DynamicImage, config: &Config) -> DynamicImage {
    let mut image = image;

    if let Some((width, height)) = config.resize {
        info!("Resizing image to {}x{}", width, height);
        image = image.resize_exact(width, height, imageops::Lanczos3);
    }

    if let Some(angle) = config.rotate {
        info!("Rotating image by {} degrees", angle);
        match angle {
            90 => image = image.rotate90(),
            180 => image = image.rotate180(),
            270 => image = image.rotate270(),
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

fn save_image(image: &DynamicImage, output_path: &Path) {
    let mut output_file = File::create(output_path).expect("Failed to create output file");
    image
        .write_to(&mut output_file, ImageFormat::Png)
        .expect("Failed to write image");
    info!("Saved image to {:?}", output_path);
}

fn read_existing_training_data(input_dir: &Path) -> HashMap<String, usize> {
    let mut label_to_class_index = HashMap::new();

    let training_data_path = input_dir.join("training_data.json");
    if training_data_path.exists() {
        let file =
            File::open(&training_data_path).expect("Failed to open existing training data file");
        let existing_data: Vec<TrainingData> =
            serde_json::from_reader(file).expect("Failed to read existing training data");
        for data in existing_data {
            label_to_class_index
                .entry(data.labels)
                .or_insert(data.class_index);
        }
    }

    label_to_class_index
}

fn process_directory(
    input_dir: &Path,
    output_dir: &Path,
    config: &Config,
    label_to_class_index: &Arc<Mutex<HashMap<String, usize>>>,
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
            if extension == Some("png") || extension == Some("jpg") || extension == Some("jpeg") {
                info!("Processing image file: {:?}", path);
                match image::open(path) {
                    Ok(image) => {
                        let processed_image = process_image(image, config);

                        let relative_path = path.strip_prefix(input_dir).expect("Invalid path");
                        let output_path = output_dir.join(relative_path);
                        if let Some(parent) = output_path.parent() {
                            fs::create_dir_all(parent).expect("Failed to create output directory");
                        }
                        save_image(&processed_image, &output_path);

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
                    }
                    Err(e) => {
                        error!("Failed to open image {:?}: {}", path, e);
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

fn main() {
    env_logger::Builder::new()
        .filter(None, LevelFilter::Info)
        .init();
    let _args = Cli::parse();

    let input_dir: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter the input directory containing images")
        .interact_text()
        .unwrap();
    let input_dir = Path::new(&input_dir);

    let output_dir: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter the output directory for modified images")
        .interact_text()
        .unwrap();
    let output_dir = Path::new(&output_dir);

    let resize: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter resize dimensions in the format WIDTHxHEIGHT (e.g., 800x600) or press Enter to skip")
        .allow_empty(true)
        .interact_text()
        .unwrap();
    let resize = if resize.is_empty() {
        None
    } else {
        Some(resize)
    };

    let rotate: String = Input::with_theme(&ColorfulTheme::default())
        .with_prompt("Enter rotate angle (90, 180, 270) or press Enter to skip")
        .allow_empty(true)
        .validate_with(|input: &String| -> Result<(), &str> {
            if input.is_empty() {
                return Ok(());
            }
            match input.parse::<u32>() {
                Ok(val) if val == 90 || val == 180 || val == 270 => Ok(()),
                _ => Err("Please enter 90, 180, or 270"),
            }
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

    let config = Config {
        resize: resize.map(|r| parse_resize(r).expect("Invalid resize format")),
        rotate,
        flip_horizontal,
        flip_vertical,
    };

    let start_time = Instant::now();

    let label_to_class_index = Arc::new(Mutex::new(read_existing_training_data(input_dir)));
    let training_data = process_directory(input_dir, output_dir, &config, &label_to_class_index);

    let duration = start_time.elapsed();

    let json_file_path = output_dir.join("training_data.json");
    let json_file = File::create(&json_file_path).expect("Failed to create JSON file");
    serde_json::to_writer_pretty(json_file, &training_data).expect("Failed to write JSON file");

    let processed_count = training_data.len();
    let avg_time_per_image = duration.as_secs_f64() / processed_count as f64;

    info!("Training data JSON file created at {:?}", json_file_path);
    info!("Processed {} images in {:?}", processed_count, duration);
    info!("Average time per image: {:.6} seconds", avg_time_per_image);
    println!("Processed {} images in {:?}", processed_count, duration);
    println!("Average time per image: {:.6} seconds", avg_time_per_image);
}
