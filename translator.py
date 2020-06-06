import requests

request_url = "https://openapi.naver.com/v1/papago/n2mt"
texts = ['it\'s okay, garfield',
        '1 sewed the stoffing back into your bear',
        'cool-looking scar there, pookman,']

for text in texts:
    headers = {"X-Naver-Client-Id": "pphSUkUVQ9iapBnJGHW5", "X-Naver-Client-Secret": "y5Xpn1KM48"}
    params = {"source": "en", "target": "ko", "text": text}
    response = requests.post(request_url, headers=headers, data=params)

    result = response.json()

    print(result)