# https://github.com/signavio/sap-sam/blob/main/src/sapsam/SignavioAuthenticator.py
# NOTICE: Changed directory of login data and url

import requests
from bef4llm.definitions import SapSamConstants

"""
def authenticate():
  
    Authenticates user at Signavio system instance and initiates session.
    Returns:
        dictionary: Session information
    
    login_url = SapSamConstansts.system_instance.value + '/p/login'
    data = {
        'name': SapSamConstansts.email.value,
        'password': SapSamConstansts.pw.value,
        'tokenonly': 'true'
    }
    if 'tenant_id' in locals():
        data['tenant'] = SapSamConstansts.tenant_id.value
    # authenticate
    login_request = requests.post(login_url, data)

    # retrieve token and session ID
    auth_token = login_request.content.decode('utf-8')
    jsesssion_ID = login_request.cookies['JSESSIONID']

    # The cookie is named 'LBROUTEID' for base_url 'editor.signavio.com'
    # and 'editor.signavio.com', and 'AWSELB' for base_url
    # 'app-au.signavio.com' and 'app-us.signavio.com'
    lb_route_ID = login_request.cookies['LBROUTEID']

    # return credentials
    return {
        'jsesssion_ID': jsesssion_ID,
        'lb_route_ID': lb_route_ID,
        'auth_token': auth_token
    }


"""


def authenticate():
    """
    Authenticates user at Signavio system instance and initiates session.
    Returns:
        dictionary: Session information
    """
    login_url = SapSamConstants.system_instance.value + '/p/login'
    data = {
        'name': SapSamConstants.email.value,
        'password': SapSamConstants.pw.value,
        'tokenonly': 'true',
        'tenant': SapSamConstants.tenant_id.value
    }

    # authenticate
    login_request = requests.post(login_url, data)

    # retrieve token and session ID
    auth_token = login_request.content.decode('utf-8')
    jsesssion_ID = login_request.cookies['JSESSIONID']

    # The cookie is named 'LBROUTEID' for base_url 'editor.signavio.com'
    # and 'editor.signavio.com', and 'AWSELB' for base_url
    # 'app-au.signavio.com' and 'app-us.signavio.com'
    lb_route_ID = login_request.cookies['LBROUTEID']

    # return credentials
    return {
        'jsesssion_ID': jsesssion_ID,
        'lb_route_ID': lb_route_ID,
        'auth_token': auth_token
    }

class SignavioAuthenticator:
    """
    Takes care of authentication against Signavio systems
    """

    # def authenticate(self):

