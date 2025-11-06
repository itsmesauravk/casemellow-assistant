import json

# Load raw product data
with open("../data/products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

cleaned_products = []

for p in products:
    productCategory = p.get("productCategory", "").strip().lower()
    productId = p.get("_id", "").strip()
    productUrl = f"http://localhost:3000/products/{productCategory}/{productId}"

    cleaned = {
        "productName": p.get("productName", "").strip(),
        "brandName": p.get("brands", {}).get("brandName", "").strip(),
        "phoneModel": p.get("phoneModel", "").strip(),
        "coverType": p.get("coverType", []),
        "productDescription": p.get("productDescription", "").strip(),
        "productPrice": p.get("productPrice", 0),
        "productCategory": p.get("productCategory", "").strip(),
        "productImage": p.get("productImage", "").strip(),
        "productUrl": productUrl
    }
    cleaned_products.append(cleaned)

# Save to new JSON file
with open("../data/cleaned_products.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_products, f, indent=4, ensure_ascii=False)

print(f" Cleaned {len(cleaned_products)} products and saved to cleaned_products.json")
