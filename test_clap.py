from laion_clap import CLAP_Module

# Initialize the CLAP model
model = CLAP_Module(enable_fusion=False)
model.load_ckpt()  # Download the default pretrained checkpoint
print("CLAP model loaded successfully!")

