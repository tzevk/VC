# ocr_api/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import pytesseract
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1: Preprocess the image
def extract_text(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )

    text = pytesseract.image_to_string(thresh)
    return text

# Step 2: Fuzzy match email and website
def fuzzy_email_and_website_detection(joined_text):
    # Normalize common OCR issues
    cleaned = joined_text.replace(" dot ", ".").replace(" dot", ".").replace("dot ", ".")
    cleaned = cleaned.replace(" at ", "@").replace("(at)", "@").replace("[at]", "@")
    cleaned = re.sub(r"[|:;,]", "", cleaned)

    # Split into chunks to isolate real emails
    words = cleaned.split()

    emails = []
    for word in words:
        # Clean up trailing punctuation
        candidate = word.strip().replace(" ", "").strip(".:;")
        match = re.match(r"[\w\.-]+@[\w\.-]+\.\w+", candidate)
        if match:
            email = match.group()
            # Remove numbers from the beginning (like in your case)
            email = re.sub(r"^\d+", "", email)
            emails.append(email)

    websites = []
    for word in words:
        if "www." in word:
            websites.append(word.strip().strip(",.;"))

    # Return only the first email and website for now
    return (emails[0] if emails else None, websites[0] if websites else None)

# Step 3: Extract structured info
def extract_info(text):
    info = {
        "name": None,
        "email": None,
        "phone": [],
        "company": None,
        "website": None,
        "address": None
    }

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    joined_text = " ".join(lines)

    email, website = fuzzy_email_and_website_detection(joined_text)
    info["email"] = email
    info["website"] = website

    phone_matches = re.findall(r'(\+?\d[\d\s\-()]{7,})', joined_text)
    info["phone"] = [p.strip() for p in phone_matches]

    for line in lines:
        if re.search(r'(pvt|technologies|solutions|inc|ltd|corp)', line.lower()):
            info["company"] = line
            break

    address_keywords = ['road', 'street', 'nagar', 'mumbai', 'india', 'lane', 'police station', 'santacruz']
    address_lines = [line for line in lines if any(k in line.lower() for k in address_keywords)]
    if address_lines:
        info["address"] = ", ".join(address_lines)

    for line in lines:
        if email and email in line:
            continue
        if any(phone in line for phone in info["phone"]):
            continue
        if website and website in line:
            continue
        if re.search(r'(pvt|solutions|tech|www|@)', line.lower()):
            continue
        if len(line.split()) >= 2 and not info["name"]:
            info["name"] = line
            break

    return info

@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.jpg", "wb") as f:
        f.write(contents)

    text = extract_text("temp.jpg")
    info = extract_info(text)
    return info
