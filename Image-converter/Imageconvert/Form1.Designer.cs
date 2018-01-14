namespace ImageConverter
{
    partial class imageViewForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.inputPictureBox = new System.Windows.Forms.PictureBox();
            this.outputPictureBox = new System.Windows.Forms.PictureBox();
            this.openImageButton = new System.Windows.Forms.Button();
            this.generateeOutputText = new System.Windows.Forms.Button();
            this.openFileButton = new System.Windows.Forms.Button();
            this.getImageButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.inputPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.outputPictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // inputPictureBox
            // 
            this.inputPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.inputPictureBox.Location = new System.Drawing.Point(49, 49);
            this.inputPictureBox.Name = "inputPictureBox";
            this.inputPictureBox.Size = new System.Drawing.Size(574, 417);
            this.inputPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.inputPictureBox.TabIndex = 0;
            this.inputPictureBox.TabStop = false;
            // 
            // outputPictureBox
            // 
            this.outputPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.outputPictureBox.Location = new System.Drawing.Point(753, 49);
            this.outputPictureBox.Name = "outputPictureBox";
            this.outputPictureBox.Size = new System.Drawing.Size(574, 417);
            this.outputPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.AutoSize;
            this.outputPictureBox.TabIndex = 1;
            this.outputPictureBox.TabStop = false;
            // 
            // openImageButton
            // 
            this.openImageButton.Location = new System.Drawing.Point(49, 620);
            this.openImageButton.Name = "openImageButton";
            this.openImageButton.Size = new System.Drawing.Size(106, 40);
            this.openImageButton.TabIndex = 2;
            this.openImageButton.Text = "Open Image";
            this.openImageButton.UseVisualStyleBackColor = true;
            this.openImageButton.Click += new System.EventHandler(this.openImageButton_Click);
            // 
            // generateeOutputText
            // 
            this.generateeOutputText.Location = new System.Drawing.Point(426, 620);
            this.generateeOutputText.Name = "generateeOutputText";
            this.generateeOutputText.Size = new System.Drawing.Size(180, 40);
            this.generateeOutputText.TabIndex = 3;
            this.generateeOutputText.Text = "Generate!!";
            this.generateeOutputText.UseVisualStyleBackColor = true;
            this.generateeOutputText.Click += new System.EventHandler(this.generateeOutputText_Click);
            // 
            // openFileButton
            // 
            this.openFileButton.Location = new System.Drawing.Point(753, 620);
            this.openFileButton.Name = "openFileButton";
            this.openFileButton.Size = new System.Drawing.Size(92, 40);
            this.openFileButton.TabIndex = 4;
            this.openFileButton.Text = "Open File";
            this.openFileButton.UseVisualStyleBackColor = true;
            this.openFileButton.Click += new System.EventHandler(this.openFileButton_Click);
            // 
            // getImageButton
            // 
            this.getImageButton.Location = new System.Drawing.Point(1204, 620);
            this.getImageButton.Name = "getImageButton";
            this.getImageButton.Size = new System.Drawing.Size(123, 40);
            this.getImageButton.TabIndex = 5;
            this.getImageButton.Text = "Get Image!!";
            this.getImageButton.UseVisualStyleBackColor = true;
            this.getImageButton.Click += new System.EventHandler(this.getImageButton_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(251, 26);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(175, 17);
            this.label1.TabIndex = 6;
            this.label1.Text = "Open Original Image First!!";
            // 
            // imageViewForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1411, 700);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.getImageButton);
            this.Controls.Add(this.openFileButton);
            this.Controls.Add(this.generateeOutputText);
            this.Controls.Add(this.openImageButton);
            this.Controls.Add(this.outputPictureBox);
            this.Controls.Add(this.inputPictureBox);
            this.Name = "imageViewForm";
            this.Text = "Image View";
            ((System.ComponentModel.ISupportInitialize)(this.inputPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.outputPictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox inputPictureBox;
        private System.Windows.Forms.PictureBox outputPictureBox;
        private System.Windows.Forms.Button openImageButton;
        private System.Windows.Forms.Button generateeOutputText;
        private System.Windows.Forms.Button openFileButton;
        private System.Windows.Forms.Button getImageButton;
        private System.Windows.Forms.Label label1;
    }
}

