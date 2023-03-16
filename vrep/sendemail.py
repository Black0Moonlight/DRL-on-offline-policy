# 能做到群发送邮件带附件！！超文本，png,txt,zip,
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders  # 转码


def send_email_by_qq(to):
    sender_mail = 'jrx0126@126.com'
    sender_pass = 'ZNQLAAQDCKRYPRNH'

    # 设置总的邮件体对象，对象类型为mixed
    msg_root = MIMEMultipart('mixed')
    # 邮件添加的头尾信息等
    msg_root['From'] = 'jrx0126@126.com<jrx0126@126.com>'
    msg_root['To'] = to
    # 邮件的主题，显示在接收邮件的预览页面
    subject = 'Training has been completed'
    msg_root['subject'] = Header(subject, 'utf-8')

    # 构造文本内容
    text_info = 'Training has been completed'
    text_sub = MIMEText(text_info, 'plain', 'utf-8')
    msg_root.attach(text_sub)

    # 构造txt附件
    txt_file = open(r'./log/train_log.txt', 'rb').read()
    txt = MIMEBase('txt', 'txt', filename='train_log.txt')
    # 以下代码可以重命名附件为测试.txt
    txt.add_header('content-disposition', 'attachment', filename=('utf-8', '', 'train_log.txt'))
    txt.add_header('Content-ID', '<0>')
    txt.add_header('X-Attachment-Id', '0')
    txt.set_payload(txt_file)
    encoders.encode_base64(txt)
    msg_root.attach(txt)
    '''
    # 构造超文本
    url = "https://blog.csdn.net/chinesepython"
    html_info = """
    <p>点击以下链接，你会去向一个更大的世界</p>
    <p><a href="%s">click me</a></p>
    <p>i am very galsses for you</p>
    """ % url
    html_sub = MIMEText(html_info, 'html', 'utf-8')
    # 如果不加下边这行代码的话，上边的文本是不会正常显示的，会把超文本的内容当做文本显示
    html_sub["Content-Disposition"] = 'attachment; filename="csdn.html"'
    # 把构造的内容写到邮件体中
    msg_root.attach(html_sub)

    # 构造图片附件
    image_file = open(r'测试图片.png', 'rb').read()
    image = MIMEBase('png', 'png', filename='测试图片.png')
    # 以下代码可以重命名附件为测试图片.png
    image.add_header('content-disposition', 'attachment', filename=('utf-8', '', '图片.png'))
    image.add_header('Content-ID', '<0>')
    image.add_header('X-Attachment-Id', '0')
    image.set_payload(image_file)
    encoders.encode_base64(image)
    msg_root.attach(image)

    

    # 构造ppt附件
    ppt_file = open(r'测试附件.ppt', 'rb').read()
    ppt = MIMEBase('ppt', 'ppt', filename='测试.ppt')
    # 以下代码可以重命名附件为测试.ppt
    ppt.add_header('content-disposition', 'attachment', filename=('utf-8', '', '测试.ppt'))
    ppt.add_header('Content-ID', '<0>')
    ppt.add_header('X-Attachment-Id', '0')
    ppt.set_payload(ppt_file)
    encoders.encode_base64(ppt)
    msg_root.attach(ppt)

    # 构造zip附件
    zip_file = open(r'先进控制仿真题目.zip', 'rb').read()
    # 这里附件的MIME和文件名，这里是zip类型
    zip = MIMEBase('zip', 'zip', filename='先进控制仿真题目.zip')
    # 加上必要的头信息
    zip.add_header('Content-Disposition', 'attachment', filename=('gb2312', '', '先进控制仿真题目.zip'))
    zip.add_header('Content-ID', '<0>')
    zip.add_header('X-Attachment-Id', '0')
    # 把附件的内容读进来
    zip.set_payload(zip_file)
    # 用Base64编码
    encoders.encode_base64(zip)
    msg_root.attach(zip)

    # 构造eml附件
    eml_file = open(r'测试附件.eml', 'rb').read()
    # 这里附件的MIME和文件名，这里是eml类型
    eml = MIMEBase('eml', 'eml', filename='测试1.eml')
    # 加上必要的头信息
    eml.add_header('Content-Disposition', 'attachment', filename=('utf-8', '', '测试附件2.eml'))
    eml.add_header('Content-ID', '<0>')
    eml.add_header('X-Attachment-Id', '0')
    # 把附件的内容读进来
    eml.set_payload(eml_file)
    # 用Base64编码
    encoders.encode_base64(eml)
    msg_root.attach(eml)

    # 构造pdf附件
    pdf_file = open(r'测试附件.pdf', 'rb').read()
    # 这里附件的MIME和文件名，这里是pdf类型
    pdf = MIMEBase('pdf', 'pdf', filename='测试1.pdf')
    # 加上必要的头信息
    pdf.add_header('Content-Disposition', 'attachment', filename=('utf-8', '', '测试附件1.pdf'))
    pdf.add_header('Content-ID', '<0>')
    pdf.add_header('X-Attachment-Id', '0')
    # 把附件的内容读进来
    pdf.set_payload(pdf_file)
    # 用Base64编码
    encoders.encode_base64(pdf)
    msg_root.attach(pdf)
    
    '''
    try:
        sftp_obj = smtplib.SMTP('smtp.126.com', 25)
        sftp_obj.login(sender_mail, sender_pass)
        sftp_obj.sendmail(sender_mail, to, msg_root.as_string())
        sftp_obj.quit()
        # print('sendemail successful!')

    except Exception as e:
        print('sendemail failed next is the reason')
        print(e)


if __name__ == '__main__':
    # 邮箱地址是一个列表，支持多个邮件地址同时发送。可以从excel文件中append地址数据到列表中(如果有需要)
    tolists = ['jrx0126@126.com', '373815350@qq.com']
    # tolists = ['jrx0126@126.com']
    total = len(tolists)
    hadsend = 0
    for tolist in tolists:
        to = tolist
        send_email_by_qq(to)
        print('had sent to||%s||successfully ———— %d email had been sent' % (tolist, hadsend + 1))
        hadsend += 1
    if total == hadsend:
        print('all emails had been sent!!Totally had sent %d already' % hadsend)
    else:
        print('fail to send all emails!!There are still %d emails do not be sent' % (total - hadsend))
