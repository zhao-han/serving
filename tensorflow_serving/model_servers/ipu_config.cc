/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_serving/model_servers/ipu_config.h"

#include "tensorflow/compiler/plugin/poplar/driver/config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace serving {

IpuConfig::IpuConfig() {
  auto platform =
      tensorflow::se::MultiPlatformManager::PlatformWithName("Poplar");
  platform_ =
      static_cast<xla::poplarplugin::PoplarPlatform*>(platform.ValueOrDie());
}

tensorflow::Status IpuConfig::ConfigIpu() {
  VLOG(1) << "Config IPU device. Number IPUs = "
            << num_ipus;
  options_.set_creator_id(xla::poplarplugin::IpuOptionsCreator::IPU_UTILS);
  auto device_config = options_.add_device_config();
  device_config->set_auto_count(num_ipus);
  return platform_->ConfigurePoplarDevices(options_);
}

}  // namespace serving
}  // namespace tensorflow