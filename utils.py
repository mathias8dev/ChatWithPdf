import uuid
  
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def generate_access_token():
    return f"key-{uuid.uuid4().hex}"

def generate_and_save_access_token():
    key = generate_access_token()
    with open("apiKey.txt", "w") as file:
        file.write(f"{key}\n")

def is_access_token_valid(token):
    with open("apiKey.txt", "r") as file:
        return file.readline().strip('\n') == token