import os
from plot_visualizer.models import User, UploadedImage
from flask import (
    Blueprint,
    render_template, 
    url_for,
    flash,
    redirect,
    request,
    current_app
)
from plot_visualizer.users.forms import(
    RegistrationForm,
    LoginForm,
    UpdateAccountForm,
    RequestResetForm,
    ResetPasswordForm,
    SubmitImageForm
)
from plot_visualizer import bcrypt, db
from flask_login import(
    login_user,
    current_user,
    logout_user,
    login_required
)
from plot_visualizer.users.utils import save_picture, send_reset_email
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from plot_visualizer.ml_utils.img_classify import classify_image


users = Blueprint('users', __name__)

@users.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now login with your account', category='success')
        return redirect(url_for('users.login'))
    return render_template('./register.html', title='Register', form=form)

@users.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            # Go to the original next page if it exists
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('main.home'))
        flash('Login unsuccessful. Please check your credentials.', category='danger')
    return render_template('./login.html', title='Login', form=form)

@users.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@users.route('/account', methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data, path='static/profile_pics')
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', category='success')
        return redirect(url_for('users.account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('./account.html', title='Account', image_file=image_file, form=form)

@users.route("/reset_password", methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password.', category='info')
        return redirect(url_for('users.login'))
    return render_template('./reset_request.html', title='Reset Password', form=form)

@users.route("/reset_password/<token>", methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    user = User.verify_reset_token(token)
    if not user:
        flash('That is an invalid or expired token.', category='warning')
        return redirect(url_for('users.reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Your password has been updated. You are now able to log in.', category='success')
        return redirect(url_for('users.login'))
    return render_template('./reset_token.html', title='Reset Password', form=form)

@users.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    form = SubmitImageForm()
    file_pt = ''
    if form.validate_on_submit():
        try:
            file = form.picture.data
            extension = os.path.splitext(file.filename)[-1].lower()
            if extension not in current_app.config['ALLOWED_EXTENSIONS']:
                flash("File is in a currently unsupported format.", category='error')
                return redirect(url_for('users.dashboard'))
            file_pt = save_picture(file, path='static/images', adjust_size=False)
            plot_type = classify_image(file)
            img = UploadedImage(image_file=file_pt, plot_type=plot_type, user_id=current_user.id)
            db.session.add(img)
            db.session.commit()
        except RequestEntityTooLarge:
            flash('File is larger than 16MB Limit', category='error')
            return redirect(url_for('users.dashboard'))
        flash('You have submitted a new image!', category='success')
        # print(UploadedImage.query.all())
        return redirect(url_for('images.analyze_image', filename=file_pt))
    
    images = UploadedImage.query.filter_by(user_id=current_user.id).all()
    image_file_paths = [image.image_file for image in images]
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('./dashboard.html', form=form, image_file=image_file, images=image_file_paths)