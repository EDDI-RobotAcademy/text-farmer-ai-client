import os
import sys

from first_user_defined_function_domain.service.fudf_service_impl import FudfServiceImpl
from first_user_defined_function_domain.service.request.fudf_just_for_test_request import FudfJustForTestRequest
from first_user_defined_function_domain.service.response.fudf_just_for_test_response import FudfJustForTestResponse
from openai_tf_idf.service.openai_tf_idf_service_impl import OpenAITfIdfServiceImpl
from openai_tf_idf.service.request.openai_tf_idf_request import OpenAITfIdfRequest
from openai_tf_idf.service.response.openai_tf_idf_response import OpenAITfIdfResponse
from tf_idf_bow.service.request.tf_idf_bow_request import TfIdfBowRequest
from tf_idf_bow.service.response.tf_idf_bow_response import TfIdfBowResponse
from tf_idf_bow.service.tf_idf_bow_service_impl import TfIdfBowServiceImpl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))

from template.custom_protocol.service.custom_protocol_service_impl import CustomProtocolServiceImpl
from template.request_generator.request_class_map import RequestClassMap
from template.response_generator.response_class_map import ResponseClassMap

from user_defined_protocol.protocol import UserDefinedProtocolNumber


class UserDefinedProtocolRegister:
    @staticmethod
    def registerDefaultUserDefinedProtocol():
        customProtocolService = CustomProtocolServiceImpl.getInstance()
        firstUserDefinedFunctionService = FudfServiceImpl.getInstance()

        requestClassMapInstance = RequestClassMap.getInstance()
        requestClassMapInstance.addRequestClass(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            FudfJustForTestRequest
        )

        responseClassMapInstance = ResponseClassMap.getInstance()
        responseClassMapInstance.addResponseClass(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            FudfJustForTestResponse
        )

        customProtocolService.registerCustomProtocol(
            UserDefinedProtocolNumber.FIRST_USER_DEFINED_FUNCTION_FOR_TEST,
            firstUserDefinedFunctionService.justForTest
        )

    # @staticmethod
    # def registerTfIdfBowProtocol():
    #     customProtocolService = CustomProtocolServiceImpl.getInstance()
    #     tfIdfBowService = TfIdfBowServiceImpl.getInstance()
    #
    #     requestClassMapInstance = RequestClassMap.getInstance()
    #     requestClassMapInstance.addRequestClass(
    #         UserDefinedProtocolNumber.FIND_SIMILAR_ANSWER,
    #         TfIdfBowRequest
    #     )
    #
    #     responseClassMapInstance = ResponseClassMap.getInstance()
    #     responseClassMapInstance.addResponseClass(
    #         UserDefinedProtocolNumber.FIND_SIMILAR_ANSWER,
    #         TfIdfBowResponse
    #     )
    #
    #     customProtocolService.registerCustomProtocol(
    #         UserDefinedProtocolNumber.FIND_SIMILAR_ANSWER,
    #         tfIdfBowService.findSimilarAnswerInfo
    #     )

    @staticmethod
    def registerOpenAITfIdfProtocol():
        customProtocolService = CustomProtocolServiceImpl.getInstance()
        openAITfIdfService = OpenAITfIdfServiceImpl.getInstance()

        requestClassMapInstance = RequestClassMap.getInstance()
        requestClassMapInstance.addRequestClass(
            UserDefinedProtocolNumber.OPENAI_TF_IDF,
            OpenAITfIdfRequest
        )

        responseClassMapInstance = ResponseClassMap.getInstance()
        responseClassMapInstance.addResponseClass(
            UserDefinedProtocolNumber.OPENAI_TF_IDF,
            OpenAITfIdfResponse
        )

        customProtocolService.registerCustomProtocol(
            UserDefinedProtocolNumber.OPENAI_TF_IDF,
            openAITfIdfService.textSimilarityAnalysis
        )

    @staticmethod
    def registerUserDefinedProtocol():
        # defalut
        UserDefinedProtocolRegister.registerDefaultUserDefinedProtocol()

        # UserDefinedProtocolRegister.registerTfIdfBowProtocol()
        UserDefinedProtocolRegister.registerOpenAITfIdfProtocol()