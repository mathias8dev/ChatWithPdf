from flask import json


class ApiResponse:

    @staticmethod
    def success(app, content):
        return app.response_class(
            response=json.dumps({
                "code": 200,
                "description": {
                    "message": "Request processed successfully",
                    "content": content
                }
            }),
            status=200,
            mimetype="application/json"
        )

    @staticmethod
    def badRequest(app, content):
        return app.response_class(
            response=json.dumps({
                "code": 400,
                "description": {
                    "message": "Bad request",
                    "error": content
                }
            }),
            status=400,
            mimetype="application/json"
        )

    @staticmethod
    def genericResponse(app, code, content):
        return app.response_class(
            response=json.dumps({
                "code": code,
                "description": content,
            }),
            status=code,
            mimetype="application/json"
        )
