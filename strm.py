import streamlit as st
import streamlit.components.v1 as stc
import GenerateFeatures
import Classification
import io
import numpy as np
import cv2
# File Processing Pkgs
import pandas as pd
#import docx2txt
from PIL import Image 
#from PyPDF2 import PdfFileReader
#import pdfplumber

def load_image(image_file):
    img = Image.open(image_file)
    return img


def grade(image_file,imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider):
    
    file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
    img = load_image(image_file)
    imgLocation.image(img,image_file.name,use_column_width='never')              
    #textLocation1.write("Processing..please wait.!")
    fw = file_details['Filename']+".jpg"
    pb = pbLocation.progress(0)
    textLocation1.info("Processing...Please wait.!")
    img_array = np.array(img)
    cv2.imwrite(fw, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    pb.progress(20)
    features = GenerateFeatures.create_features_set(fw)
    pb.progress(50)
    probs = Classification.predict(features)
    pb.progress(80)
    #st.write(probs)
    gradesdic = {}
    for i in range(len(probs)):
        if(probs.loc[i,"class"]==0):
            label = "Normal Tissue"
        elif(probs.loc[i,"class"]==1):
            label = "Grade 3"
        elif(probs.loc[i,"class"]==2):
            label = "Grade 4"
        elif(probs.loc[i,"class"]==3):
            label = "Grade 5"
        prob = probs.loc[i,"probs"]*100
        
        if(i==0):
            textLocation2.markdown("Prediction for Class : __*"+label+"*__, Score : "+"__{:.2f}__".format(prob)+"%")
            gradesdic[1] = label
            pb.progress(85)
        elif(i==1):
            textLocation3.write("Prediction for Class : "+label+", Score : "+"{:.2f}".format(prob)+"%")
            gradesdic[2] = label
            pb.progress(90)
        elif(i==2):
            textLocation4.write("Prediction forClass : "+label+", Score : "+"{:.2f}".format(prob)+"%")
            gradesdic[3] = label
            pb.progress(95)
        elif(i==3):
            textLocation5.write("Prediction for Class : "+label+", Score : "+"{:.2f}".format(prob)+"%")
            gradesdic[4] = label
            pb.progress(100)
        pbLocation.empty()
        textLocation1.success("Grading has done successfully")
    sectionDivider.markdown("____")
    colg1, colg2= st.columns(2)
    gradetosave = colg1.text_input("Grade to Save", gradesdic[1])
    
    subcol1,subcol2,subcol3 = st.columns([1,1,10])
    save = subcol2.button("Save")
    if  save:
        st.write(gradetosave)
    return gradetosave


def main():
    st.title("Prostate Cancer Grading System")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home")
        image_files= st.file_uploader("Upload Image",type=['jpeg','jpg'],accept_multiple_files=True)
        if image_files :
            
            textLocation1 = st.empty()
            textLocation2 = st.empty()
            textLocation3 = st.empty()
            textLocation4 = st.empty()
            textLocation5 = st.empty()
            col1, col2, col3,col4= st.columns(4)
            with col1:
                    imgLocation = st.empty()
            with col3:
                if(len(image_files)!=1):
                    nxtbtnLocation = st.empty()
                    prvbtnLocation = st.empty()
                    nxtbtn = nxtbtnLocation.button("Next Image >>")
                    st.text('')
                    prvbtn = prvbtnLocation.button("<< Previous Image")
            pbLocation = st.empty()
            textLocation1 = st.empty()
            txcol1,txcol2 ,txcol3= st.columns([1,3,1])
            with txcol2:
                    textLocation2 = st.empty()
                    textLocation3 = st.empty()
                    textLocation4 = st.empty()
                    textLocation5 = st.empty()
            sectionDivider = st.empty()
            if 'i' not in st.session_state:
                st.session_state.i=0
                gradetosave=grade(image_files[0],imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider)
            
            if(len(image_files)!=1):
                if nxtbtn:
                    st.session_state.i += 1
                    #st.write(st.session_state.i)
                    if(st.session_state.i<len(image_files)-1):
                        gradetosave=grade(image_files[st.session_state.i],imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider)
                    elif(st.session_state.i==len(image_files)-1):
                        gradetosave = grade(image_files[st.session_state.i],imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider)
                        nxtbtnLocation.write("No images to load further")
                elif prvbtn:
                    st.session_state.i -= 1
                    if(st.session_state.i>0):
                        gradetosave = grade(image_files[st.session_state.i],imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider)
                    elif(st.session_state.i==0):
                        gradetosave = grade(image_files[st.session_state.i],imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5,pbLocation,sectionDivider)
                        prvbtnLocation.write("No images to go back further")
                
                    
        
            #for image_file in image_files:
             #   grade(image_file,imgLocation,textLocation1,textLocation2,textLocation3,textLocation4,textLocation5)

            # To See Details
            #st.write(type(image_file))
            #st.write(dir(image_file))
            #st.write(file_details)
           
    elif (choice=="About"):
        st.subheader("About")
        st.write("\n")
        st.markdown("__Developer __: Samal Kaveesha.")
        st.write("This system was developed to grade the prostate cancers using texture feature analysis. Features were calculated by converting the original image into gray image and the three color channels. This system uses svm with a rbf kernel for making predictions. This system has an accuracy of 93% with 400 set of  training images. You can download sample images from the following link to test the system.")
        st.write("URL for sample images : https://github.com/Samal-Kavz/Cancer-Grading-using-texture-features/tree/main/Sample_Images")
        st.markdown("__How the system works ? __")
        st.write("You can upload images one by one or multiple images at once. After making the prediction the each class is shown from maximum probability to lowest. If images are uploaded one by one, after uploading the new image press 'Next Image' to grade the newly uploaded image. If you upload multiple images at once you can use 'Next Image' and 'Previous Image' buttons to go between images.")
        st.write("__Note : Whole slide images can not  be graded using this version.Please use tiled images of whole slide images.New version will come soon with the feature of grading whole slide images.__")
            

if __name__ == '__main__':
    main()

