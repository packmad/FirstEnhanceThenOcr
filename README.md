# FirstEnhanceThenOcr

Forensic image-enhancement + OCR pipeline


### Dependencies

```
sudo apt install tesseract-ocr

pip3 install -r requirements.txt

wget -O model/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```


### Usage example

```
python3 enhance_ocr.py example.jpeg -o case42
```
