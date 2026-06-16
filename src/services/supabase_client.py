from globals import SUPABASE_URL, SUPABASE_KEY
import supabase

supabase_client = None

def initialize_supabase_client():
    global supabase_client
    if supabase_client is None:
        supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Initialized Supabase client :", supabase_client)
    return supabase_client