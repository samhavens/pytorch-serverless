import os
import boto3
import numpy as np
import urllib.request


s3_client = boto3.client('s3')


def download_file(bucket_name, object_key_name, file_path):
	"""  Downloads a file from an S3 bucket.
	:param bucket_name: S3 bucket name
	:param object_key_name: S3 object key name
	:param file_path: path to save downloaded file
	"""
	s3_client.download_file(bucket_name, object_key_name, file_path)


def get_labels(path):
	"""  Get labels from a text file.
	:param path: path to text file
	:return: list of labels
	"""
	with open(path, encoding='utf-8', errors='ignore') as f:
		labels = [line.strip() for line in f.readlines()]
		f.close()
	return labels
