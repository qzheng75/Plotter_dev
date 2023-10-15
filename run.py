from plot_visualizer import create_app, db
from plot_visualizer.models import UploadedImage

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        db.session.query(UploadedImage).delete()
        db.session.commit()
    app.run(debug=True)
