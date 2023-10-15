from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import(
    FileField,
    SubmitField,
)
from flask_wtf.file import FileField, FileRequired, FileAllowed

class ImageSubmitForm(FlaskForm):
    file = FileField('File', validators=[
        FileRequired(),  # File is required
        FileAllowed(['jpg', 'jpeg', 'png'], 'Only image files allowed.')
    ])
    submit = SubmitField('Submit Image')