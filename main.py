import http.client
from utils import fill_average_generated_prob, read_applications
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == "__main__":
    applications_file = "application.csv" # EDIT THIS
    field_name = "explanation" # EDIT THIS
    final_file_name = f"evaluated/{applications_file}_with_{field_name}_evaluated.csv" # don't need to edit this but you can if you want to

    
    # print(os.getenv("GPTZERO_API_KEY"))
    df = fill_average_generated_prob(read_applications(file_path=applications_file), field_name)
    df.to_csv(final_file_name, index=False)