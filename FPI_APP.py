import os
import sqlite3
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Determine project base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Configuration
DATABASE_NAME = os.path.join(BASE_DIR, 'fingerprints.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
TEMPLATES_FOLDER = os.path.join(BASE_DIR, 'templates')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_EMPLOYEES = 5

CONFIG = {
    'image_size': 96,
    'batch_size': 64,
    'embedding_dim': 128,
    'margin': 0.6,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 7,
    'data_dir': '/kaggle/working/Soco_Altered_Combined',
    'model_save_path': '/kaggle/working/Models_combined',
    'seed': 42
}
# Ensure templates directory and basic templates exist BEFORE Flask app instantiation
def init_templates():
    os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
    index_path = os.path.join(TEMPLATES_FOLDER, 'index.html')
    register_path = os.path.join(TEMPLATES_FOLDER, 'register.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Fingerprint App</title></head>
<body>
  <h1>Fingerprint Registration</h1>
  <p>{{ count }} of {{ max_employees }} employees registered.</p>
  <a href="{{ url_for('register') }}">Register New Employee</a>
</body>
</html>''')
    if not os.path.exists(register_path):
        with open(register_path, 'w', encoding='utf-8') as f:
            f.write('''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Register Employee</title></head>
<body>
  <h1>Register New Employee</h1>
  <form method="post" enctype="multipart/form-data">
    <label>Name: <input type="text" name="name" required></label><br>
    <label>Fingerprint: <input type="file" name="fingerprint" accept="image/*" required></label><br>
    <button type="submit">Register</button>
  </form>
  <p><a href="{{ url_for('index') }}">Back to Home</a></p>
</body>
</html>''')

# Create templates before creating the app
init_templates()

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder=TEMPLATES_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'

# Initialize database only if it doesn't exist
def init_db():
    if not os.path.exists(DATABASE_NAME):
        conn = sqlite3.connect(DATABASE_NAME)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

# Helper: check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load trained model once
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=CONFIG['embedding_dim']):
        super(SiameseNetwork, self).__init__()
        
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Second conv block with attention
        self.conv2_main = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.skip_conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.att2 = AttentionBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        
        # Third conv block with attention
        self.conv3_main = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.skip_conv3 = nn.Conv2d(64, 128, kernel_size=1)
        self.att3 = AttentionBlock(128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Fourth conv block with attention
        self.conv4_main = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.skip_conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.att4 = AttentionBlock(256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Calculate output dimensions
        conv_output_size = CONFIG['image_size'] // 16
        self.conv_output_dim = 256 * conv_output_size * conv_output_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    
    def forward_one(self, x):
        # First block
        x = self.conv1(x)
        
        # Second block with attention
        x_main = self.conv2_main(x)
        x_skip = self.skip_conv2(x)
        x = F.relu(x_main + x_skip)
        x = self.att2(x)
        x = self.pool2(x)
        
        # Third block with attention
        x_main = self.conv3_main(x)
        x_skip = self.skip_conv3(x)
        x = F.relu(x_main + x_skip)
        x = self.att3(x)
        x = self.pool3(x)
        
        # Fourth block with attention
        x_main = self.conv4_main(x)
        x_skip = self.skip_conv4(x)
        x = F.relu(x_main + x_skip)
        x = self.att4(x)
        x = self.pool4(x)
        
        # Flatten and pass through FC layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, anchor, positive, negative):
        anchor_out = self.forward_one(anchor)
        positive_out = self.forward_one(positive)
        negative_out = self.forward_one(negative)
        return anchor_out, positive_out, negative_out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(embedding_dim=CONFIG['embedding_dim'])
model.load_state_dict(torch.load(os.path.join(BASE_DIR, '/home/mohammad/Desktop/Computer_Vision/Vision_Advanced/Exercises/Project_1/FP_Iden_Best.pth'), map_location=device))
model.eval()

# Image preprocessing
def preprocess_image(path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((CONFIG['img_height'], CONFIG['img_width'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(path).convert('L')
    return transform(image).unsqueeze(0).to(device)

# Route: Home
@app.route('/')
def index():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM employees')
    count = c.fetchone()[0]
    conn.close()
    return render_template('index.html', count=count, max_employees=MAX_EMPLOYEES)

# Route: Register new employee
@app.route('/register', methods=['GET', 'POST'])
def register():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM employees')
    count = c.fetchone()[0]
    if count >= MAX_EMPLOYEES:
        flash(f'Maximum of {MAX_EMPLOYEES} employees reached.')
        conn.close()
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form['name']
        file = request.files['fingerprint']
        if not name or not file or not allowed_file(file.filename):
            flash('Invalid name or file type.')
            conn.close()
            return redirect(request.url)

        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Generate embedding
        img_tensor = preprocess_image(save_path)
        with torch.no_grad():
            embedding = model.forward_one(img_tensor).cpu().numpy()

        # Store in DB
        c.execute('INSERT INTO employees (name, embedding) VALUES (?, ?)',
                  (name, embedding.tobytes()))
        conn.commit()
        conn.close()
        flash('Employee registered successfully.')
        return redirect(url_for('index'))

    conn.close()
    return render_template('register.html')

# Run initialization and app
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5100, debug=True)
