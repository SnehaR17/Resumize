
###### Packages Used ######
import streamlit as st # core package used in this project
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
import nltk
nltk.download('stopwords')

from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,web_course,android_course,ios_course,uiux_course,resume_videos,interview_videos

from linkedin_api import Linkedin

# from dotenv import load_dotenv
# import toml

# # Load secrets from .env.toml
# secrets = toml.load(".env.toml")["secrets"]

# Set env variables (optional if your platform injects automatically)
# os.environ["DB_HOST"] = secrets["DB_HOST"]
# os.environ["DB_USER"] = secrets["DB_USER"]
# os.environ["DB_PASSWORD"] = secrets["DB_PASSWORD"]
# os.environ["DB_NAME"] = secrets["DB_NAME"]



###### Preprocessing functions ######


# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format 
def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    ## bytes conversions
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    ## close open handles
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######


# sql connector
# connection = pymysql.connect(host='localhost',user='root',password='03jan@1978#',db='cv')
connection = pymysql.connect(
    host=st.secrets["mysql"]["host"],
    user=st.secrets["mysql"]["user"],
    password=st.secrets["mysql"]["password"],
    db=st.secrets["mysql"]["database"]
)

cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token,ip_add,host_name,dev_user,os_name_ver,latlong,city,state,country,act_name,act_mail,act_mob,name,email,res_score,timestamp,no_of_pages,reco_field,cand_level,skills,recommended_skills,courses,pdf_name):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (str(sec_token),str(ip_add),host_name,dev_user,os_name_ver,str(latlong),city,state,country,act_name,act_mail,act_mob,name,email,str(res_score),timestamp,str(no_of_pages),reco_field,cand_level,skills,recommended_skills,courses,pdf_name)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="Resumize | AI Resume Analyzer",
   page_icon='./Logo/RESUM.png',
)


import requests


def fetch_linkedin_jobs(reco_field, location="India"):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_jobs",
        "q": reco_field,
        "location": location,
        "api_key": st.secrets["api"]["serpapi_key"]  
    }

    response = requests.get(url, params=params)
    results = response.json()

    print(results["jobs_results"][0].keys())

    # jobs = []
    # if "jobs_results" in results:
    #     for job in results["jobs_results"]:
    #         job_title = job.get("title")
    #         job_link = job.get("link")
    #         if job_title and job_link:
    #             jobs.append(f"{job_title} - {job_link}")
    return results["jobs_results"][0:5]


###### Main function run() ######


def run():
    
    # (Logo, Heading, Sidebar etc)
    img = Image.open('./Logo/RESUM.png')
    st.image(img)
    st.sidebar.markdown("# Choose Something...")
    activities = ["User", "Feedback", "About", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '<b>Built by Sneha R, Nikhil Kumar, Simha Harshith and Abhinav Patil</b>' 
    st.sidebar.markdown(link, unsafe_allow_html=True)

    ###### Creating Database and Table ######


    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)


    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)


    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)


    ###### CODE FOR CLIENT SIDE (USER) ######

    if choice == 'User':
        
        # Collecting Miscellaneous Information
        act_name = st.text_input('Name*', placeholder="Enter your Name")
        act_mail = st.text_input('Mail*', placeholder="Enter your mail")
        act_mob  = st.text_input('Mobile Number*', placeholder="Enter your mobile number")
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng
        geolocator = Nominatim(user_agent="http")
        location = geolocator.reverse(latlong, language='en')
        address = location.raw['address']
        cityy = address.get('city', '')
        statee = address.get('state', '')
        countryy = address.get('country', '')  
        city = cityy
        state = statee
        country = countryy


        # Upload Resume
        st.markdown('''<h5 style='text-align: left; color: #C3AEFF;'> Upload Your Resume, And Get Smart Recommendations</h5>''',unsafe_allow_html=True)
        
        ## file upload in pdf format
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            ### saving the uploaded resume to folder
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            
            if resume_data:
                
                ## Get the whole resume data into resume_text
                resume_text = pdf_reader(save_image_path)

                ## Showing Analyzed data from (resume_data)
                st.header("**Resume Analysis 🤘**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info 👀**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))

                except:
                    pass
                ## Predicting Candidate Experience Level 

                import re
                text = resume_text.lower()
                pages = resume_data.get('no_of_pages', 0)
                cand_level = ''
                if pages == 0:
                    cand_level = "NA"
                    st.markdown(
                        "<h4 style='text-align: left; color: #d73b5c;'>No resume detected (NA)</h4>",
                        unsafe_allow_html=True
                    )

                # 2) ≤1 page, no mention of internships → Fresher
                elif pages <= 1 and 'internship' not in text:
                    cand_level = "Fresher"
                    st.markdown(
                        "<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>",
                        unsafe_allow_html=True
                    )

                # 3) Any mention of “internship”
                elif 'internship' in text:
                    cand_level = "Intermediate"
                    st.markdown(
                        "<h4 style='text-align: left; color: #1ed760;'>You are at Intermediate level!</h4>",
                        unsafe_allow_html=True
                    )

                # 4) A true “experience” section header — match whole word, optionally preceded by “work”
                elif re.search(r'\b(work\s+)?experience\b', text):
                    cand_level = "Experienced"
                    st.markdown(
                        "<h4 style='text-align: left; color: #fba171;'>You are at Experienced level!</h4>",
                        unsafe_allow_html=True
                    )

                # 5) Fallback → Fresher, with the correct “Fresher” colour
                else:
                    cand_level = "Fresher"
                    st.markdown(
                        "<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>",
                        unsafe_allow_html=True
                    )

                ## Skills Analyzing and Recommendation
                st.subheader("**Skills Recommendation 💡**")
                
                ### Current Analyzed Skills
                keywords = st_tags(label='### Your Current Skills',
                text='See our skills recommendation below',value=resume_data['skills'],key = '1  ')

                ds_keyword = [
                # Core ML/DL frameworks & libraries
                'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'mlflow',
                # Data manipulation & analysis
                'pandas', 'numpy', 'scipy', 'dask', 'vaex',
                # Data viz & reporting
                'matplotlib', 'seaborn', 'plotly', 'bokeh', 'dash', 'ggplot',
                # Specialized AI
                'computer vision', 'opencv', 'natural language processing', 'nlp', 'spaCy', 'nltk', 'transformers', 'bert', 'gpt',
                # Statistical & classical methods
                'regression', 'classification', 'clustering', 'time series', 'ARIMA', 'bayesian inference', 'hypothesis testing',
                # Advanced topics
                'reinforcement learning', 'deep reinforcement learning', 'rnn', 'lstm', 'gru', 'cnn', 'gan', 'autoencoder',
                # Data platforms & tooling
                'hadoop', 'spark', 'hive', 'kafka', 'airflow', 'kubernetes', 'docker', 'bigquery', 'redshift',
                # General DS concepts
                'feature engineering', 'hyperparameter tuning', 'cross validation', 'model deployment', 'mLOps', 'data pipeline'
                ]

                web_keyword = [
                    # Frontend frameworks & libraries
                    'react', 'react js', 'vue', 'angular js', 'svelte', 'ember.js', 'jquery',
                    # Backend frameworks & languages
                    'django', 'flask', 'express.js', 'node js', 'asp.net', 'spring boot', 'ruby on rails', 'php', 'laravel', 'magento',
                    # Full-stack ecosystems
                    'next.js', 'nuxt.js', 'gatsby', 'meteor',
                    # Web fundamentals
                    'html5', 'css3', 'sass', 'less', 'javascript', 'typescript',
                    # APIs & data exchange
                    'rest api', 'graphql', 'apollo', 'soap',
                    # DevOps & hosting
                    'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'netlify', 'heroku', 'firebase',
                    # CMS & e-commerce
                    'wordpress', 'drupal', 'joomla', 'shopify', 'woocommerce',
                    # Testing & quality
                    'jest', 'mocha', 'chai', 'jasmine', 'cypress', 'selenium',
                    # Build tools & bundlers
                    'webpack', 'rollup', 'parcel', 'gulp', 'grunt'
                ]

                android_keyword = [
                    'android', 'android development', 'android sdk', 'android studio', 'java', 'kotlin', 'flutter', 'react native',
                    'jetpack compose', 'xml layouts', 'material design', 'gradle', 'retrofit', 'okhttp', 'room', 'liveData', 'viewModel',
                    'mvvm', 'mvp', 'ndk', 'jni', 'firebase', 'google play services', 'arcore', 'jetpack', 'constraint layout'
                ]

                ios_keyword = [
                    'ios', 'ios development', 'xcode', 'swift', 'objective-c', 'swiftui', 'uikit', 'cocoa', 'cocoa touch',
                    'interface builder', 'testflight', 'core data', 'core animation', 'core graphics', 'core ml', 'arkit', 'spritekit',
                    'scenekit', 'mvvm', 'vipER', 'combine', 'swift package manager', 'cocoapods', 'carthage', 'autolayout', 'storyboard'
                ]

                uiux_keyword = [
                    # Design tools
                    'figma', 'sketch', 'adobe xd', 'invision', 'axure rp', 'balsamiq', 'zeplin', 'miro', 'principle',
                    # Visual design
                    'adobe photoshop', 'photoshop', 'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects',
                    'adobe premiere pro', 'premiere pro', 'adobe indesign', 'indesign',
                    # UX fundamentals
                    'user research', 'user persona', 'user journey', 'information architecture', 'interaction design', 'accessibility',
                    'usability testing', 'heuristic evaluation', 'wireframes', 'prototyping', 'storyboards', 'user flows',
                    # UI patterns & systems
                    'design system', 'atomic design', 'responsive design', 'typography', 'color theory', 'iconography', 'style guide',
                    # Collaboration & workflow
                    'design thinking', 'agile ux', 'lean ux', 'wireframe', 'high fidelity mockup', 'low fidelity prototype', 'empathy map'
                ]

                n_any = [
                    'english', 'communication', 'writing', 'public speaking', 'presentation', 'microsoft office', 'excel', 'powerpoint',
                    'leadership', 'teamwork', 'project management', 'agile', 'scrum', 'kanban', 'time management', 'critical thinking',
                    'problem solving', 'customer management', 'stakeholder management', 'social media', 'seo', 'sem', 'digital marketing',
                    'research', 'data analysis', 'negotiation', 'mentoring'
                ]


                ### Skill Recommendations Starts                
                ### Skill Recommendations Starts                
                # import re

                # Normalize
                resume_skills_cleaned = [s.lower().replace(' ', '') for s in resume_data['skills']]
                resume_text_cleaned = resume_text.lower()

                # Define domains
                domain_keywords = {
                    'Data Science': ds_keyword,
                    'Web Development': web_keyword,
                    'Android Development': android_keyword,
                    'IOS Development': ios_keyword,
                    'UI-UX Development': uiux_keyword,
                }

                domain_courses = {
                    'Data Science': ds_course,
                    'Web Development': web_course,
                    'Android Development': android_course,
                    'IOS Development': ios_course,
                    'UI-UX Development': uiux_course,
                }

                # Match scoring
                domain_scores = {}
                for domain, keywords in domain_keywords.items():
                    score = 0
                    for kw in keywords:
                        kw_clean = kw.lower().replace(' ', '')
                        # Skill match (exact cleaned)
                        if kw_clean in resume_skills_cleaned:
                            score += 2  # stronger signal
                        # Text match (partial)
                        elif re.search(r'\b' + re.escape(kw.lower()) + r'\b', resume_text_cleaned):
                            score += 1
                    domain_scores[domain] = score

                # Pick best scoring domain
                reco_field = max(domain_scores, key=domain_scores.get)
                score = domain_scores[reco_field]

                # UI
                if score == 0:
                    st.warning("**We couldn't confidently predict your field. Try adding more technical keywords to your resume.**")
                    reco_field = 'NA'
                    recommended_skills = ['No Recommendations']
                    rec_course = "Sorry! Not Available for this Field"
                    st_tags(label='### Recommended skills for you.', text='Currently No Recommendations', value=recommended_skills, key='reco_none')
                else:
                    st.success(f"**Our analysis says you are looking for {reco_field} Jobs.**")
                    # Recommend skills not in resume
                    all_resume_tokens = set(resume_skills_cleaned)
                    all_resume_tokens.update(re.findall(r'\w+', resume_text_cleaned))

                    recommended_skills = [
                        kw for kw in domain_keywords[reco_field]
                        if kw.lower().replace(' ', '') not in all_resume_tokens
                    ]

                    st_tags(label='### Recommended skills for you.',
                            text='Recommended skills generated from System',
                            value=recommended_skills,
                            key=f'reco_{reco_field.lower().replace(" ", "_")}')

                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 the chances of getting a Job💼</h5>''',
                                unsafe_allow_html=True)

                    rec_course = course_recommender(domain_courses[reco_field])




                ## Resume Scorer & Resume Writing Tips
                st.subheader("**Resume Tips & Ideas 🥂**")


                text = resume_text.lower()

                # 2) Define your rubrics in one place
                sections = {
                    'objective_summary': {
                        'keywords': ['objective', 'summary'],
                        'weight': 6,
                        'label': "Objective/Summary",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added an Objective or Summary.",
                        'fail': "Please add your career objective or summary—this orients recruiters to your goals."
                    },
                    'education': {
                        'keywords': ['education', 'school', 'college', 'degree'],
                        'weight': 12,
                        'label': "Education",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Education details.",
                        'fail': "Please list your education to show your qualification level."
                    },
                    'experience': {
                        # Use regex to match “experience” as a whole word or “work experience”
                        'regex': [r'\bexperience\b', r'\bwork experience\b'],
                        'weight': 16,
                        'label': "Experience",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Work Experience.",
                        'fail': "Please add work experience to stand out from the crowd."
                    },
                    'internships': {
                        'keywords': ['internship', 'internships'],
                        'weight': 6,
                        'label': "Internships",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Internships.",
                        'fail': "Please add internships to demonstrate hands-on learning."
                    },
                    'skills': {
                        'keywords': ['skills', 'skill'],
                        'weight': 7,
                        'label': "Skills",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Skills.",
                        'fail': "Please add skills to show what you bring to the table."
                    },
                    'hobbies': {
                        'keywords': ['hobbies', 'hobby'],
                        'weight': 4,
                        'label': "Hobbies",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Hobbies.",
                        'fail': "Please add hobbies to reveal your personality."
                    },
                    'interests': {
                        'keywords': ['interests', 'interest'],
                        'weight': 5,
                        'label': "Interests",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Interests.",
                        'fail': "Please add interests to showcase passions beyond work."
                    },
                    'achievements': {
                        'keywords': ['achievements', 'achievement'],
                        'weight': 13,
                        'label': "Achievements",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Achievements.",
                        'fail': "Please add achievements to highlight your accomplishments."
                    },
                    'certifications': {
                        'keywords': ['certifications', 'certification'],
                        'weight': 12,
                        'label': "Certifications",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Certifications.",
                        'fail': "Please add certifications to show your specializations."
                    },
                    'projects': {
                        'keywords': ['projects', 'project'],
                        'weight': 19,
                        'label': "Projects",
                        'colour': '#1ed760',
                        'success': "Awesome! You have added Projects.",
                        'fail': "Please add projects to demonstrate relevant work."
                    }
                }

                # 3) Score and render
                resume_score = 0
                max_score = sum(info['weight'] for info in sections.values())

                for key, info in sections.items():
                    found = False

                    # Keyword-based check
                    if 'keywords' in info:
                        for kw in info['keywords']:
                            if kw in text:
                                found = True
                                break

                    # Regex-based check
                    elif 'regex' in info:
                        for pattern in info['regex']:
                            if re.search(pattern, text):
                                found = True
                                break

                    # Update score & display
                    if found:
                        resume_score += info['weight']
                        st.markdown(
                            f"<h5 style='text-align:left; color:{info['colour']}'>[+] {info['success']}</h5>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<h5 style='text-align:left; color:#fff'>[-] {info['fail']}</h5>",
                            unsafe_allow_html=True
                        )

                # 4) Final tally
                st.markdown(
                    f"<h4 style='text-align:left;'>Your resume score: <strong>{resume_score}</strong> / {max_score}</h4>",
                    unsafe_allow_html=True
                )

                st.subheader("**Resume Score 📝**")
                
                st.markdown(
                    """
                    <style>
                        .stProgress > div > div > div > div {
                            background-color: #d73b5c;
                        }
                    </style>""",
                    unsafe_allow_html=True,
                )

                ### Score Bar
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                ### Score
                st.success('** Your Resume Writing Score: ' + str(score)+'**')
                st.warning("** Note: This score is calculated based on the content that you have in your Resume. **")

                ### Getting Current Date and Time
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)


                ## Calling insert_data to add all the data into user_data                
                insert_data(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']), str(recommended_skills), str(rec_course), pdf_name)

                ## Recommending Resume Writing Video
                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                ## Recommending Interview Preparation Video
                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                # Linkedin Jobs
                st.header("**🔍 Recommended Jobs for You**")
                job_data = fetch_linkedin_jobs(reco_field)

                if not job_data:
                    st.warning("No job recommendations available.")
                    return

                for job in job_data:
                    title = job.get("title", "Unknown Title")
                    company = job.get("company_name", "Unknown Company")
                    location = job.get("location", "Unknown Location")
                    link = job.get("share_link", "#")

                    st.markdown(f"**{title}** at *{company}*  \n📍 {location}  \n🔗 [Apply Now]({link})")
                    st.markdown("---")

                # print("\n-----------------------------------------------------------------------------\n")
                
                # print(jobs)
                # print("\n-----------------------------------------------------------------------------\n")


                ## On Successful Result 
                # st.balloons()

            else:
                st.error('Something went wrong..')                


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':   
        
        # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        # Feedback Form
        with st.form("my_form"):
            st.write("Feedback form")            
            feed_name = st.text_input('Name')
            feed_email = st.text_input('Email')
            feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
            comments = st.text_input('Comments')
            Timestamp = timestamp        
            submitted = st.form_submit_button("Submit")
            if submitted:
                ## Calling insertf_data to add dat into user feedback
                insertf_data(feed_name,feed_email,feed_score,comments,Timestamp)    
                ## Success Message 
                st.success("Thanks! Your Feedback was recorded.") 
                ## On Successful Submit
                # st.balloons()    


        # query to fetch data from user feedback table
        query = 'select * from user_feedback'        
        plotfeed_data = pd.read_sql(query, connection)                        


        # fetching feed_score from the query and getting the unique values and total value count 
        labels = plotfeed_data.feed_score.unique()
        values = plotfeed_data.feed_score.value_counts()


        # plotting pie chart for user ratings
        st.subheader("**Past User Rating's**")
        fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5", color_discrete_sequence=px.colors.sequential.Aggrnyl)
        st.plotly_chart(fig, False)


        #  Fetching Comment History
        cursor.execute('select feed_name,feed_score, comments from user_feedback')
        plfeed_cmt_data = cursor.fetchall()

        st.subheader("**User Comment's**")
        dff = pd.DataFrame(plfeed_cmt_data, columns=['User','Rating', 'Comment'])
        st.dataframe(dff, width=2000)

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':   

        st.subheader("**About The Tool - AI RESUME ANALYZER**")

        st.markdown('''

        <p align='justify'>
            A tool which parses information from a resume using natural language processing and finds the keywords, cluster them onto sectors based on their keywords. And lastly show recommendations, predictions, analytics to the applicant based on keyword matching.
        </p>

        <p align="justify">
            <b>How to use it: -</b> <br/><br/>
            <b>User -</b> <br/>
            In the Side Bar choose yourself as user and fill the required fields and upload your resume in pdf format.<br/>
            Just sit back and relax our tool will do the magic on it's own.<br/><br/>
            <b>Feedback -</b> <br/>
            A place where user can suggest some feedback about the tool.<br/><br/>
            <b>Admin -</b> <br/>
            For login use <b>admin</b> as username and <b>admin@resume-analyzer</b> as password.<br/>
            It will load all the required stuffs and perform analysis.
        </p><br/><br/>

        <p align="justify">
            Built with 🤍 by 
            Sneha R, Nikhil Kumar, Simha Harshith and Abhinav Patil <br>
            under guidance of Prof. Aparna H D
        </p>

        ''',unsafe_allow_html=True)  


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    else:
        st.success('Welcome to Admin Side')

        #  Admin Login
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            
            ## Credentials 
            if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':
                
                ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, ip_add, resume_score, convert(Predicted_Field using utf8), convert(User_level using utf8), city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['Idt', 'IP_add', 'resume_score', 'Predicted_Field', 'User_Level', 'City', 'State', 'Country'])
                
                ### Total Users Count with a Welcome Message
                values = plot_data.Idt.count()
                st.success("Welcome Team ! Total %d " % values + " User's Have Used Our Tool : )")                
                
                ### Fetch user data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, sec_token, ip_add, act_name, act_mail, act_mob, convert(Predicted_Field using utf8), Timestamp, Name, Email_ID, resume_score, Page_no, pdf_name, convert(User_level using utf8), convert(Actual_skills using utf8), convert(Recommended_skills using utf8), convert(Recommended_courses using utf8), city, state, country, latlong, os_name_ver, host_name, dev_user from user_data''')
                data = cursor.fetchall()                

                st.header("**User's Data**")
                df = pd.DataFrame(data, columns=['ID', 'Token', 'IP Address', 'Name', 'Mail', 'Mobile Number', 'Predicted Field', 'Timestamp',
                                                 'Predicted Name', 'Predicted Mail', 'Resume Score', 'Total Page',  'File Name',   
                                                 'User Level', 'Actual Skills', 'Recommended Skills', 'Recommended Course',
                                                 'City', 'State', 'Country', 'Lat Long', 'Server OS', 'Server Name', 'Server User',])
                
                ### Viewing the dataframe
                st.dataframe(df)
                
                ### Downloading Report of user_data in csv file
                st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)

                ### Fetch feedback data from user_feedback(table) and convert it into dataframe
                cursor.execute('''SELECT * from user_feedback''')
                data = cursor.fetchall()

                st.header("**User's Feedback Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])
                st.dataframe(df)

                ### query to fetch data from user_feedback(table)
                query = 'select * from user_feedback'
                plotfeed_data = pd.read_sql(query, connection)                        

                ### Analyzing All the Data's in pie charts

                # fetching feed_score from the query and getting the unique values and total value count 
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                
                # Pie chart for user ratings
                st.subheader("**User Rating's**")
                fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5 🤗", color_discrete_sequence=px.colors.sequential.Aggrnyl)
                st.plotly_chart(fig)

                # fetching Predicted_Field from the query and getting the unique values and total value count                 
                labels = plot_data.Predicted_Field.unique()
                values = plot_data.Predicted_Field.value_counts()

                # Pie chart for predicted field recommendations
                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills 👽', color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
                st.plotly_chart(fig)

                # fetching User_Level from the query and getting the unique values and total value count                 
                labels = plot_data.User_Level.unique()
                values = plot_data.User_Level.value_counts()

                # Pie chart for User's👨‍💻 Experienced Level
                st.subheader("**Pie-Chart for User's Experienced Level**")
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart 📈 for User's 👨‍💻 Experienced Level", color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig)

                # fetching resume_score from the query and getting the unique values and total value count                 
                labels = plot_data.resume_score.unique()                
                values = plot_data.resume_score.value_counts()

                # Pie chart for Resume Score
                st.subheader("**Pie-Chart for Resume Score**")
                fig = px.pie(df, values=values, names=labels, title='From 1 to 100 💯', color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig)

                # fetching IP_add from the query and getting the unique values and total value count 
                labels = plot_data.IP_add.unique()
                values = plot_data.IP_add.value_counts()

                # Pie chart for Users
                st.subheader("**Pie-Chart for Users App Used Count**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On IP Address 👥', color_discrete_sequence=px.colors.sequential.matter_r)
                st.plotly_chart(fig)

                # fetching City from the query and getting the unique values and total value count 
                labels = plot_data.City.unique()
                values = plot_data.City.value_counts()

                # Pie chart for City
                st.subheader("**Pie-Chart for City**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based On City 🌆', color_discrete_sequence=px.colors.sequential.Jet)
                st.plotly_chart(fig)

                # fetching State from the query and getting the unique values and total value count 
                labels = plot_data.State.unique()
                values = plot_data.State.value_counts()

                # Pie chart for State
                st.subheader("**Pie-Chart for State**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on State 🚉', color_discrete_sequence=px.colors.sequential.PuBu_r)
                st.plotly_chart(fig)

                # fetching Country from the query and getting the unique values and total value count 
                labels = plot_data.Country.unique()
                values = plot_data.Country.value_counts()

                # Pie chart for Country
                st.subheader("**Pie-Chart for Country**")
                fig = px.pie(df, values=values, names=labels, title='Usage Based on Country 🌏', color_discrete_sequence=px.colors.sequential.Purpor_r)
                st.plotly_chart(fig)

            ## For Wrong Credentials
            else:
                st.error("Wrong ID & Password Provided")

# Calling the main (run()) function to make the whole process run
run()
