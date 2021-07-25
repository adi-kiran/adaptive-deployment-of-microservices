require "socket"
math.randomseed(socket.gettime()*1000)
math.random(); math.random(); math.random()

request = function()
  local method = "POST"
  -- Use your cluster-ip here:
  local path = "http://127.0.0.1:8082/api/sqlquery"
  local headers = {}
  headers["Content-Type"] = "application/json;charset=UTF-8"
  local body = [[ {"IP":""} ]]
  return wrk.format(method, path, headers, body)
end