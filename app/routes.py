from flask import request, jsonify
from app import app
from werkzeug.utils import secure_filename
import uuid
from app.receipt_ocr import image_enhancement, predict, ocr

# Mock data for demonstration purposes
BASE_URL = ""
API_KEY = "1"

@app.route('/api/computervision/es/receipt-ocr', methods=['POST'])
def receipt_ocr():
    # Check if file is passed
    if 'file' not in request.files:
        return jsonify({
            "code": "400 BAD_REQUEST",
            "message": "Request is Empty Please Enter Mandatory Parameters",
            "responseJson": None,
            "detailMessage": None,
            "id": None
        })

    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({
            "code": "400 BAD_REQUEST",
            "message": "No File in the request",
            "responseJson": None,
            "detailMessage": None,
            "id": None
        })

    # Check file format
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({
            "code": "400 BAD_REQUEST",
            "message": "Uploaded file format is not supported. Please upload files only in jpg, jpeg, png format",
            "responseJson": None,
            "detailMessage": None,
            "id": None
        })

    uuid4 = str(uuid.uuid4())

    # Save file
    filename = secure_filename(file.filename)
    file.save(f"app/static/images/uploads/{filename}")
    image_path = f"app/static/images/uploads/{filename}"
    enhanced_image_path = f"app/data/enhanced_{uuid4}.jpg"
    image_enhancement(image_path, enhanced_image_path)
    labelled_images_path = predict(enhanced_image_path, uuid4)
    data = ocr(labelled_images_path)
    print("\n\ndata: ", data)

    # Check if Michelob SKU is detected
    michelob_detected = any('michelob' in item['item'].lower() for item in data['items'])
    
    # Prepare response based on scenarios
    if michelob_detected:
        response_data = {
            "code": "200 OK",
            "message": "Success",
            "responseJson": {
                "merchantName": data.get('store_name'),
                "merhcantAddress": data.get('store_address'),
                "transactionDate": data.get('transaction_date'),
                "transactionTime": data.get('transaction_time'),
                "invoiceNumber": data.get('invoice_number'),
                "products": data.get('items'),
                "message": ""
            },
            "detailMessage": None,
            "id": None
        }
    else:
        response_data = {
            "code": "200 OK",
            "message": "Success",
            "responseJson": {
                "merchantName": data.get('store_name'),
                "merhcantAddress": data.get('store_address'),
                "transactionDate": data.get('transaction_date'),
                "transactionTime": data.get('transaction_time'),
                "invoiceNumber": data.get('invoice_number'),
                "products": [
                    None
                ],
                "message": "Please scan a receipt with the Michelob products you've purchased"
            },
            "detailMessage": None,
            "id": None
        }
    # elif not data['items']:
    #     response_data = {
    #         "code": "200 OK",
    #         "message": "Success",
    #         "responseJson": {
    #             "merchantName": data.get('store_name'),
    #             "merhcantAddress": data.get('store_address'),
    #             "transactionDate": data.get('transaction_date'),
    #             "transactionTime": data.get('transaction_time'),
    #             "products": [None],
    #             "message": "Please scan a receipt with the Michelob products you've purchased"
    #         },
    #         "detailMessage": None,
    #         "id": None
    #     }
    # elif not data['store_name'] and not data['store_address'] and not data['transaction_date'] and not data['transaction_time']:
    #     response_data = {
    #         "code": "200 OK",
    #         "message": "Success",
    #         "responseJson": {
    #             "merchantName": None,
    #             "merhcantAddress": None,
    #             "transactionDate": None,
    #             "transactionTime": None,
    #             "products": [None],
    #             "message": "Image is not clear"
    #         },
    #         "detailMessage": None,
    #         "id": None
    #     }
    # else:
    #     response_data = {
    #         "code": "200 OK",
    #         "message": "Success",
    #         "responseJson": {
    #             "merchantName": None,
    #             "merhcantAddress": None,
    #             "transactionDate": None,
    #             "transactionTime": None,
    #             "products": [None],
    #             "message": "Please Scan a Valid Receipt Image"
    #         },
    #         "detailMessage": None,
    #         "id": None
    #     }
    
    return jsonify(response_data)
