from flask import(
    Blueprint,
    current_app,
    send_from_directory,
    render_template
)
from flask_login import(
    login_required
)
from plot_visualizer.models import UploadedImage

images = Blueprint('images', __name__)

@images.route('/serve-image/<filename>', methods=['GET'])
@login_required
def serve_image(filename):
  return send_from_directory(current_app.config['UPLOAD_DIRECTORY'], filename)

@images.route('/analyze-image/<filename>', methods=['GET', 'POST'])
@login_required
def analyze_image(filename):
  plot = UploadedImage.query.filter_by(image_file=filename).first()
  return render_template('./analyze_image.html', image=filename, plot_type=plot.plot_type.replace('_', ' '))

@images.route('/button_clicked/<button_id>', methods=['POST'])
def button_clicked(button_id):
    # Handle the button click here
    return f'Button {button_id} clicked.'