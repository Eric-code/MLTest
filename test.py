#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from urllib import request
from bs4 import BeautifulSoup
import time
if __name__ == "__main__":
    file = open('将夜2.txt', 'w', encoding='utf-8')
    target_url = 'http://www.dzxs.cc/read/144.html'
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166  Safari/535.19'
    target_req = request.Request(url=target_url, headers=head)
    target_response = request.urlopen(target_req)
    target_html = target_response.read().decode('gbk', 'ignore')
    # 创建BeautifulSoup对象
    listmain_soup = BeautifulSoup(target_html, 'lxml')
    # 搜索文档树,找出div标签中class为chapterlist的所有子标签
    chapters = listmain_soup.find_all('dl', class_='chapterlist')
    # 使用查询结果再创建一个BeautifulSoup对象,对其继续进行解析
    download_soup = BeautifulSoup(str(chapters), 'lxml')
    chap = download_soup.find_all('dd')
    download_so = BeautifulSoup(str(chap), 'lxml')
    # print(download_so.body.contents[3].a.string)
    # 开始记录内容标志位,只要正文卷下面的链接,最新章节列表链接剔除
    begin_flag = True
    x = 0
    # 遍历dl标签下所有子节点
    for child in download_so.body.contents:
        x = x + 1
        if child != '，' and x % 2 == 0:
            # print(child.a.string)
            # 爬取链接
            if begin_flag == True and child.a != None:
                download_url = "http://www.dzxs.cc" + child.a.get('href')
                download_req = request.Request(url=download_url, headers=head)
                download_response = request.urlopen(download_req)
                download_html = download_response.read().decode('gbk', 'ignore')
                download_name = child.a.string
                soup_texts = BeautifulSoup(download_html, 'lxml')
                texts = soup_texts.find_all(id='BookText')
                soup_text = BeautifulSoup(str(texts), 'lxml')
                write_flag = True
                file.write(download_name + '\n\n')
                # 将爬取内容写入文件
                for each in soup_text.div.text.replace('\xa0', ''):
                    if each == 'h':
                        write_flag = False
                    if write_flag == True and each != ' ':
                        file.write(each)
                    if write_flag == True and each == '\r':
                        file.write('\n')
                file.write('\n')
                time.sleep(1)
                # print(child.a.string)
                # print(download_name + " : " + download_url)
