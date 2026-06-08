import pickle

model = pickle.load(open("model_sms_spam.pkl", "rb"))

print(model)
print(hasattr(model, "classes_"))