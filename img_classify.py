from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import pipeline
from PIL import Image
import os


# Pretrained model and processor information
img_classify_repo_name = "qzheng75/swin-tiny-patch4-window7-224-finetuned-plot-images"
img_classify_processor = AutoImageProcessor.from_pretrained(img_classify_repo_name)
img_classify_model = AutoModelForImageClassification.from_pretrained(img_classify_repo_name)
img_classify_pipe = pipeline("image-classification",
                            model=img_classify_model,
                            feature_extractor=img_classify_processor)
is_plot_repo_name = "qzheng75/swin-tiny-patch4-window7-224-finetuned-image-is-plot-or-not-finetuned-image-is-plot-or-not"
is_plot_processor = AutoImageProcessor.from_pretrained(is_plot_repo_name)
is_plot_model = AutoModelForImageClassification.from_pretrained(is_plot_repo_name)
is_plot_pipe = pipeline("image-classification",
                        model=is_plot_model,
                        feature_extractor=is_plot_processor)

def classify_image(img_path):
    """
    Classify an image using a pretrained model.

    Args:
        img_path (str): Path to the input image file.

    Returns:
        str: If the input isn't a plot, return 'is_not_plot'. Else, return the predicted label for the image: line, scatter, dot, vertical_bar, or horizontal_bar.
    """
    image = Image.open(img_path)
    is_plot_pred = is_plot_pipe(image)
    is_plot = max(is_plot_pred, key=lambda x: x['score'])
    if is_plot['label'] == 'is_not_plot':
        return 'is_not_plot'
    class_predictions = img_classify_pipe(image)
    result = max(class_predictions, key=lambda x: x['score'])
    return result['label']

if __name__ == '__main__':
    # Example usage:
    for f in os.listdir('./test_images'):
        print(f"File {f}: {classify_image(os.path.join('test_images', f))}")