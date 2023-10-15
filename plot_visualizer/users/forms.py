from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import(
    StringField,
    PasswordField,
    SubmitField,
    BooleanField,
)
from wtforms.validators import(
    DataRequired,
    Length,
    Email,
    EqualTo,
    ValidationError
)
from plot_visualizer.models import User
from flask_login import current_user
from plot_visualizer.ml_utils.img_classify import classify_image


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), Length(min=2, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired()
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), EqualTo('password')
    ])
    submit = SubmitField('Sign Up')
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('The username has been taken. Please enter another username.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('The email has been taken. Please enter another email.')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired()
    ])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')


class UpdateAccountForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), Length(min=2, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    picture = FileField('Update Profile Picture', validators=[
        FileAllowed(['jpg', 'png'])
    ])
    submit = SubmitField('Update')
    def validate_username(self, username):
        if username.data != current_user.username:
            user = User.query.filter_by(username=username.data).first()
            if user:
                raise ValidationError('The username has been taken. Please enter another username.')
    
    def validate_email(self, email):
        if email.data != current_user.email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError('The email has been taken. Please enter another email.')

class RequestResetForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    submit = SubmitField('Request Password Reset')
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if not user:
            raise ValidationError('There is no account with this email. Please register first.')
    
class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[
        DataRequired()
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), EqualTo('password')
    ])
    submit = SubmitField('Reset Password')

class SubmitImageForm(FlaskForm):
    picture = FileField('Click the button to submit a file. (Currently only\
                        .jpg, .png, .jpeg files with size < 16MB are supported)', validators=[
        DataRequired(),
        FileAllowed(['jpg', 'png', 'jpeg'])
    ])
    submit = SubmitField('Submit Plot for analysis')

    def validate_picture(self, picture):
        res = classify_image(picture.data)
        if res == 'is_not_plot':
            raise ValidationError('The input image doesn\'t seem to be a plot. Check the content of your image or try submitting another image')