using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace ImageConverter
{
    public partial class imageViewForm : Form
    {
        public imageViewForm()
        {
            InitializeComponent();
            inputImage = null;
        }
        UInt32[] data;
        public Bitmap inputImage;
        private void ConvertRGB2Grey(Bitmap inputImage)
        {
            try
            {
                var dlg = new FolderBrowserDialog();
                dlg.Description = "Set the Path for the data of the given file";
                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    using (StreamWriter sr = new StreamWriter(dlg.SelectedPath + "\\data.txt", false))
                    {
                        for (int i = 0; i < inputImage.Width; i++)
                        {
                            for (int j = 0; j < inputImage.Height; j++)
                            {
                                Color c = inputImage.GetPixel(i, j);
                                int grey = (int)((c.R * 0.3) + (c.G * 0.59) + (c.B * 0.11));
                                sr.Write(grey.ToString() + "\n");
                            }
                        }
                    }
                    using (StreamWriter sr = new StreamWriter(dlg.SelectedPath + "\\size.txt", false))
                    {
                        sr.Write(inputImage.Width.ToString() + " " + inputImage.Height.ToString());
                    }
                }
            }
            catch (IOException ie)
            {
                MessageBox.Show(ie.Message, "Error!!");
            }
            
                
            
        }
        private void openImageButton_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Title = "Open Image";
            if(dlg.ShowDialog()==DialogResult.OK)
            {
                inputImage = new Bitmap(dlg.FileName);
                Bitmap displayImage = GenerateDisplayImage(inputImage);
                inputPictureBox.Image = displayImage;
            }
            dlg.Dispose();
        }

        private void generateeOutputText_Click(object sender, EventArgs e)
        {
            if (inputImage == null)
            {
                MessageBox.Show("Error", "Input Original Image First!!!");
                return;
            }
            ConvertRGB2Grey(inputImage);
        }

        private void openFileButton_Click(object sender, EventArgs e)
        {
            if (inputImage == null)
            {
                MessageBox.Show("Error", "Input Original Image First!!!");
                return;
            }
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Title = "Open File";
            if(inputImage==null)
            {
                MessageBox.Show("Error", "Input Original Image First!!!");
                return;
            }
            data = new UInt32[inputImage.Width * inputImage.Height];

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                StreamReader sr = new StreamReader(dlg.FileName);
                try
                {

                }
                catch (Exception)
                {

                    throw;
                }
                for (int i = 0; i < inputImage.Width; i++)
                {
                    for (int j = 0; j < inputImage.Height; j++)
                    {
                        string character = sr.ReadLine();
                        
                        data[i * inputImage.Height + j] = Convert.ToUInt32(character);
                    }
                }
            }
            dlg.Dispose();
        }

        private void getImageButton_Click(object sender, EventArgs e)
        {
            if (inputImage == null)
            {
                MessageBox.Show("Error", "Input Original Image First!!!");
                return;
            }
            Bitmap outputImageGray = new Bitmap(inputImage);
            for (int i = 0; i < inputImage.Width; i++)
            {
                for (int j = 0; j < inputImage.Height; j++)
                {
                    int datatmp = (int)data[i * inputImage.Height + j];
                    outputImageGray.SetPixel(i, j, Color.FromArgb(datatmp, datatmp, datatmp));
                }
            }
            outputPictureBox.Image = GenerateDisplayImage(outputImageGray);
        }

        private Bitmap GenerateDisplayImage(Bitmap inputImage)
        {
            float ratio = inputImage.Width / (float)inputImage.Height;
            int newWidth, newHeight;
            if (ratio > 1.5)
            {
                newWidth = 600;
                newHeight = (int)(newWidth / ratio);
            }
            else
            {
                newHeight = 400;
                newWidth = (int)(newHeight * ratio);
            }
            Bitmap displayImage = new Bitmap(inputImage, new Size(newWidth, newHeight));
            return displayImage;
        }
    }
}
